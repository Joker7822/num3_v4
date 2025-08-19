# -*- coding: utf-8 -*-
"""
Mizuho銀行 ナンバーズ3 抽せん結果スクレイパ
- JSで後差しされるため、中身(textContent)が埋まるまで待つ
- 1回分=1テーブル(.js-lottery-temp-pc)単位で安全に抽出
- 既存CSVに追記（同一「抽せん日」をスキップ）＆日付昇順で整列

必要:
  pip install selenium pandas
  Chrome/Chromium + 対応するChromeDriver(またはwebdriver-managerで自動化可)
"""

import csv
import os
import re
from datetime import datetime

import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


URL = "https://www.mizuhobank.co.jp/takarakuji/check/numbers/numbers3/index.html"
CSV_PATH = "numbers3.csv"


def build_driver(headless: bool = True) -> webdriver.Chrome:
    options = Options()
    # 新ヘッドレスの方が描画互換が高い
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1280,2400")  # PC レイアウトを確実に
    # UA を明示（ヘッドレス検出/モバイル振り分けを避ける）
    options.add_argument(
        "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
    driver = webdriver.Chrome(options=options)
    return driver


def wait_until_loaded(driver, timeout: int = 25):
    wait = WebDriverWait(driver, timeout)
    # 「表示に時間がかかっております…」の非表示を待つ（無い場合は即OK）
    try:
        wait.until(EC.invisibility_of_element_located((By.CSS_SELECTOR, ".js-now-loading")))
    except Exception:
        pass

    # 最低限のテーブル存在を待機
    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, ".section__table-wrap .js-lottery-temp-pc")))

    # 先頭テーブルの主要フィールドが埋まるまで待つ
    def first_table_ready(driver_):
        tables_ = driver_.find_elements(By.CSS_SELECTOR, ".section__table-wrap .js-lottery-temp-pc")
        if not tables_:
            return False
        t = tables_[0]
        try:
            date_el = t.find_element(By.CSS_SELECTOR, ".js-lottery-date-pc")
            issue_el = t.find_element(By.CSS_SELECTOR, ".js-lottery-issue-pc")
            num_el = t.find_element(By.CSS_SELECTOR, ".js-lottery-number-pc")
        except Exception:
            return False

        def filled(el):
            return el is not None and el.get_attribute("textContent").strip() != ""

        return filled(date_el) and filled(issue_el) and filled(num_el)

    wait.until(first_table_ready)


def parse_tables(driver):
    tables = driver.find_elements(By.CSS_SELECTOR, ".section__table-wrap .js-lottery-temp-pc")
    results = []

    for idx, t in enumerate(tables, start=1):
        try:
            date_el = t.find_element(By.CSS_SELECTOR, ".js-lottery-date-pc")
            issue_el = t.find_element(By.CSS_SELECTOR, ".js-lottery-issue-pc")
            num_el = t.find_element(By.CSS_SELECTOR, ".js-lottery-number-pc")
        except Exception:
            # テーブル形状が違うor未完成
            continue

        # textContent 抽出
        date_text = date_el.get_attribute("textContent").strip()
        issue_text = issue_el.get_attribute("textContent").strip()
        num_text = num_el.get_attribute("textContent").strip()

        # 空・見出しのテーブルはスキップ
        if not (date_text and issue_text and re.search(r"\d", num_text or "")):
            continue

        # 日付パース（例: 2025年08月18日 → 2025-08-18）
        try:
            draw_date = datetime.strptime(date_text, "%Y年%m月%d日").strftime("%Y-%m-%d")
        except Exception:
            # 予期せぬ形式はスキップ
            continue

        # 数字（全角混在対策で \d を抽出）
        digits = [int(d) for d in re.findall(r"\d", num_text)]
        main_number = str(digits)

        # 賞金: tr.js-lottery-prize-pc の最後の td 内 strong.section__text--bold
        prize_rows = t.find_elements(By.CSS_SELECTOR, "tr.js-lottery-prize-pc")
        prizes = []
        for pr in prize_rows:
            strongs = pr.find_elements(By.CSS_SELECTOR, "td strong.section__text--bold, td strong")
            val = None
            if strongs:
                txt = strongs[-1].get_attribute("textContent").strip()
                if "円" in txt:
                    try:
                        val = int(txt.replace(",", "").replace("円", "").strip())
                    except Exception:
                        val = None
            prizes.append(val)

        # 期待は5種類：ストレート/ボックス/セット(スト)/セット(ボ)/ミニ
        while len(prizes) < 5:
            prizes.append(None)

        results.append(
            {
                "回別": issue_text,
                "抽せん日": draw_date,
                "本数字": main_number,
                "ストレート": prizes[0],
                "ボックス": prizes[1],
                "セット(ストレート)": prizes[2],
                "セット(ボックス)": prizes[3],
                "ミニ": prizes[4],
            }
        )

    return results


def load_existing(csv_path: str):
    try:
        df = pd.read_csv(csv_path)
        # dtypeブレ防止のため str 化
        df["抽せん日"] = df["抽せん日"].astype(str)
        existing_dates = set(df["抽せん日"].tolist())
        fieldnames = df.columns.tolist()
        return df, existing_dates, fieldnames
    except FileNotFoundError:
        fieldnames = [
            "抽せん日",
            "本数字",
            "回別",
            "ストレート",
            "ボックス",
            "セット(ストレート)",
            "セット(ボックス)",
            "ミニ",
        ]
        return pd.DataFrame(), set(), fieldnames


def append_and_sort(csv_path: str, fieldnames, new_rows):
    if not new_rows:
        return 0

    write_header = (not os.path.exists(csv_path)) or os.path.getsize(csv_path) == 0
    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(new_rows)

    df = pd.read_csv(csv_path)
    # 安全のため日付を文字列→datetimeにしてから並べ替え（失敗時は文字列のまま）
    try:
        df["抽せん日_dt"] = pd.to_datetime(df["抽せん日"], errors="coerce")
        df.sort_values(["抽せん日_dt", "回別"], inplace=True)
        df.drop(columns=["抽せん日_dt"], inplace=True)
    except Exception:
        df.sort_values(["抽せん日", "回別"], inplace=True)

    df.to_csv(csv_path, index=False, encoding="utf-8")
    return len(new_rows)


def main():
    driver = build_driver(headless=True)
    try:
        driver.get(URL)
        wait_until_loaded(driver, timeout=25)
        scraped = parse_tables(driver)
    finally:
        driver.quit()

    if not scraped:
        print("[INFO] 新規に取得できる行が見つかりませんでした（ページ構造変更/描画待ち不足の可能性）。")
        return

    existing_df, existing_dates, fieldnames = load_existing(CSV_PATH)
    # 同一「抽せん日」を除外（必要なら回別もキーに含める）
    new_rows = [row for row in scraped if row["抽せん日"] not in existing_dates]

    saved = append_and_sort(CSV_PATH, fieldnames, new_rows)

    if saved:
        print(f"[INFO] {saved}件を保存し、日付順に並び替えました。CSV: {CSV_PATH}")
    else:
        print("[INFO] 既存CSVと同一日のため、新規保存はありません。")

    # 結果表示
    for row in new_rows:
        print(row)


if __name__ == "__main__":
    main()
