import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import csv

LOG_FILE = "scraping_log.txt"
CSV_FILE = "numbers3.csv"

def log(msg):
    print(msg)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

url = "https://www.mizuhobank.co.jp/takarakuji/check/numbers/numbers3/index.html"

try:
    response = requests.get(url, timeout=10)
    response.encoding = "utf-8"
    soup = BeautifulSoup(response.text, "html.parser")
    log("[INFO] ページ取得成功")

    tables = soup.select("table.typeTK")
    rows = tables[0].select("tr")[1:]  # ヘッダ除外

    data = []
    for row in rows:
        cols = row.select("td")
        if len(cols) < 5:
            continue
        try:
            draw_number = cols[0].text.strip()
            draw_date = datetime.strptime(cols[1].text.strip(), "%Y年%m月%d日").strftime("%Y-%m-%d")
            main_number = str([int(d) for d in cols[2].text.strip()])

            prize_values = [td.text.strip().replace(",", "").replace("円", "") for td in cols[3:]]
            prize_values = [int(p) if p.isdigit() else None for p in prize_values]

            record = {
                "回別": draw_number,
                "抽せん日": draw_date,
                "本数字": main_number,
                "ストレート": prize_values[0],
                "ボックス": prize_values[1],
                "セット(ストレート)": prize_values[2],
                "セット(ボックス)": prize_values[3],
                "ミニ": prize_values[4],
            }
            data.append(record)
            log(f"[DATA] {record}")
        except Exception as e:
            log(f"[WARNING] パースエラー: {e}")

    # 保存
    try:
        existing = pd.read_csv(CSV_FILE)
        existing_dates = existing["抽せん日"].tolist()
        fieldnames = existing.columns.tolist()
    except FileNotFoundError:
        existing = pd.DataFrame()
        existing_dates = []
        fieldnames = ["抽せん日", "本数字", "回別", "ストレート", "ボックス", "セット(ストレート)", "セット(ボックス)", "ミニ"]

    new_rows = [row for row in data if row["抽せん日"] not in existing_dates]
    if new_rows:
        with open(CSV_FILE, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if existing.empty:
                writer.writeheader()
            writer.writerows(new_rows)

        df = pd.read_csv(CSV_FILE)
        df.sort_values("抽せん日", inplace=True)
        df.to_csv(CSV_FILE, index=False, encoding="utf-8")
        log(f"[INFO] {len(new_rows)}件を保存し、日付順に並び替えました。")
    else:
        log("[INFO] 新規データはありませんでした。")

except Exception as e:
    log(f"[ERROR] スクレイピング中に例外発生: {e}")
