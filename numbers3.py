import csv
import os
import re
import time
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# === ✅ Chromeドライバー設定 ===
chrome_options = Options()
# chrome_options.add_argument("--headless")  # ← テスト時は表示確認のためコメントアウト
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument('--ignore-certificate-errors')
chrome_options.add_argument('--disable-dev-shm-usage')
chrome_service = Service()

driver = webdriver.Chrome(service=chrome_service, options=chrome_options)

# === ✅ 出力ファイル設定 ===
csv_file = "numbers3.csv"
fieldnames = ["抽せん日", "本数字", "回別"]

# === ✅ 重複チェック用のキー集合 ===
existing_keys = set()
if os.path.exists(csv_file):
    with open(csv_file, newline='', encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["抽せん日"], row["本数字"])
            existing_keys.add(key)

def append_to_csv_unique(row):
    key = (row["抽せん日"], row["本数字"])
    if key in existing_keys:
        return False
    with open(csv_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if f.tell() == 0:
            writer.writeheader()
        writer.writerow(row)
        existing_keys.add(key)
    return True

# === ✅ URLリスト ===
old_urls = [f"https://www.mizuhobank.co.jp/takarakuji/check/numbers/backnumber/num{str(i).zfill(4)}.html" for i in range(1, 2682, 20)]

new_urls = []
for start in range(2701, 6458 + 1, 20):
    end = min(start + 19, 6458)
    new_urls.append(f"https://www.mizuhobank.co.jp/takarakuji/check/numbers/backnumber/detail.html?fromto={start}_{end}&type=numbers")

month_urls = []
for year in range(2024, 2026):
    for month in range(1, 13):
        if year == 2025 and month > 4:
            break
        url = f"https://www.mizuhobank.co.jp/takarakuji/check/numbers/numbers3/index.html?year={year}&month={month}"
        month_urls.append(url)

# === ✅ 旧形式 ===
for url in old_urls:
    driver.get(url)
    time.sleep(2)
    saved_count = 0
    try:
        rows = driver.find_elements(By.CSS_SELECTOR, "table.section__table.pc-only tr.section__table-row")
        for row in rows:
            cells = row.find_elements(By.CSS_SELECTOR, "p.section__text")
            if len(cells) >= 3:
                draw_number = cells[0].text.strip()
                draw_date = cells[1].text.strip()
                main_number = cells[2].text.strip()
                if not re.fullmatch(r"\d{3}", main_number):
                    continue
                main_number = str([int(d) for d in main_number])  # ← 追加
                draw_date = datetime.strptime(draw_date, "%Y年%m月%d日").strftime("%Y-%m-%d")
                saved = append_to_csv_unique({
                    "回別": draw_number,
                    "抽せん日": draw_date,
                    "本数字": main_number
                })
                if saved:
                    saved_count += 1
                    print(f"  → 保存: {draw_date} {main_number} {draw_number}")
        print(f"[旧形式読込] {url}（新規保存: {saved_count} 件）")
    except Exception as e:
        print(f"[旧形式エラー] {url} : {e}")

# === ✅ 新形式（fromto 対応＋stale 対策）===
for url in new_urls:
    driver.get(url)
    saved_count = 0
    try:
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "tr.js-lottery-backnumber-temp-pc"))
        )
        rows = driver.find_elements(By.CSS_SELECTOR, "tr.js-lottery-backnumber-temp-pc")
        for i in range(len(rows)):
            try:
                row = driver.find_elements(By.CSS_SELECTOR, "tr.js-lottery-backnumber-temp-pc")[i]  # stale防止
                draw_number = row.find_element(By.CSS_SELECTOR, "th.section__table-head").text.strip()
                cells = row.find_elements(By.CSS_SELECTOR, "td.section__table-data")
                if len(cells) < 2:
                    continue
                draw_date_raw = cells[0].text.strip()
                main_number = cells[1].text.strip()
                if not draw_date_raw or not re.match(r"\d{4}年\d{1,2}月\d{1,2}日", draw_date_raw):
                    continue
                if not re.fullmatch(r"\d{3}", main_number):
                    continue
                main_number = str([int(d) for d in main_number])  # ← 追加
                draw_date = datetime.strptime(draw_date_raw, "%Y年%m月%d日").strftime("%Y-%m-%d")
                if append_to_csv_unique({
                    "回別": draw_number,
                    "抽せん日": draw_date,
                    "本数字": main_number
                }):
                    saved_count += 1
                    print(f"  → 保存: {draw_date} {main_number} {draw_number}")
            except Exception as e_inner:
                print(f"    [行スキップ] {e_inner}")
        print(f"[新形式読込] {url}（新規保存: {saved_count} 件）")
    except Exception as e:
        print(f"[新形式エラー] {url} : {e}")

# === ✅ 月別形式 ===
for url in month_urls:
    driver.get(url)
    time.sleep(2)
    saved_count = 0
    try:
        tables = driver.find_elements(By.CSS_SELECTOR, "table.section__table")
        for table in tables:
            try:
                draw_number = table.find_element(By.CSS_SELECTOR, "tr:nth-of-type(1) th:nth-of-type(2)").text.strip()
                draw_date = table.find_element(By.CSS_SELECTOR, "tr:nth-of-type(2) td").text.strip()
                main_number = table.find_element(By.CSS_SELECTOR, "tr:nth-of-type(3) td b.js-lottery-number-pc").text.strip()
                if not re.fullmatch(r"\d{3}", main_number):
                    continue
                main_number = str([int(d) for d in main_number])
                draw_date = datetime.strptime(draw_date, "%Y年%m月%d日").strftime("%Y-%m-%d")

                # === ✅ 賞金情報の取得 ===
                prizes = table.find_elements(By.CSS_SELECTOR, "tr.js-lottery-prize-pc strong.section__text--bold")
                prize_labels = ["ストレート", "ボックス", "セット（ストレート）", "セット（ボックス）", "ミニ"]
                prize_data = {}
                for label, prize in zip(prize_labels, prizes):
                    prize_data[label] = prize.text.replace(",", "").replace("円", "").strip()

                row = {
                    "回別": draw_number,
                    "抽せん日": draw_date,
                    "本数字": main_number
                }

                # CSV拡張：賞金を含める場合はこのように（必要に応じて）
                row.update({
                    "ストレート": prize_data.get("ストレート", ""),
                    "ボックス": prize_data.get("ボックス", ""),
                    "セット（ストレート）": prize_data.get("セット（ストレート）", ""),
                    "セット（ボックス）": prize_data.get("セット（ボックス）", ""),
                    "ミニ": prize_data.get("ミニ", "")
                })

                saved = append_to_csv_unique(row)
                if saved:
                    saved_count += 1
                    print(f"  → 保存: {draw_date} {main_number} {draw_number}")
            except Exception as e_inner:
                print(f"    [行スキップ] {e_inner}")
        print(f"[月別形式読込] {url}（新規保存: {saved_count} 件）")
    except Exception as e:
        print(f"[月別形式エラー] {url} : {e}")

# === ✅ 最新抽せん結果（トップページ）取得 ===
latest_url = "https://www.mizuhobank.co.jp/takarakuji/check/numbers/numbers3/index.html"
driver.get(latest_url)
time.sleep(2)

latest_data = []

try:
    wait = WebDriverWait(driver, 10)
    dates = wait.until(EC.presence_of_all_elements_located((By.CLASS_NAME, "js-lottery-date-pc")))
    numbers = driver.find_elements(By.CLASS_NAME, "js-lottery-number-pc")
    issues = driver.find_elements(By.CLASS_NAME, "js-lottery-issue-pc")
    prize_elems = driver.find_elements(By.CSS_SELECTOR, "tr.js-lottery-prize-pc strong.section__text--bold")

    num_results = min(len(dates), len(numbers), len(issues))

    for i in range(num_results):
        try:
            draw_date = datetime.strptime(dates[i].text.strip(), "%Y年%m月%d日").strftime("%Y-%m-%d")
            draw_number = issues[i].text.strip()
            main_number = str([int(d) for d in numbers[i].text.strip()])

            base_index = i * 5
            def get_prize(j):
                try:
                    return prize_elems[base_index + j].text.replace(",", "").replace("円", "").strip()
                except:
                    return ""

            row = {
                "回別": draw_number,
                "抽せん日": draw_date,
                "本数字": main_number,
                "ストレート": get_prize(0),
                "ボックス": get_prize(1),
                "セット（ストレート）": get_prize(2),
                "セット（ボックス）": get_prize(3),
                "ミニ": get_prize(4),
            }

            if append_to_csv_unique(row):
                print(f"  → 保存: {draw_date} {main_number} {draw_number}")

        except Exception as e:
            print(f"[WARNING] 回 {i+1} でエラー: {e}")

except Exception as e:
    print(f"[最新取得エラー] {e}")

driver.quit()

# === ✅ ここから追加 ===
print("[INFO] データを日付順に並び替え中...")
if os.path.exists(csv_file):
    with open(csv_file, newline='', encoding="utf-8") as f:
        reader = csv.DictReader(f)
        sorted_rows = sorted(reader, key=lambda row: row["抽せん日"])

    with open(csv_file, "w", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted_rows)
    print("[INFO] 並び替え完了！")
