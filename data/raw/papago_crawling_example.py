from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import time
import os

# ▶️ ChromeDriver 경로
CHROMEDRIVER_PATH = os.getenv("CHROMEDRIVER_PATH")
if CHROMEDRIVER_PATH is None:
    raise ValueError("Please set CHROMEDRIVER_PATH as environment variable.")

# ▶️ Selenium 옵션 설정
chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")

# ▶️ WebDriver 실행
driver = webdriver.Chrome(service=Service(CHROMEDRIVER_PATH), options=chrome_options)

# ▶️ Papago 크롤링 함수
def get_papago_outputs(japanese_text):
    driver.get("https://papago.naver.com")
    try:
        input_box = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "textarea#txtSource"))
        )
        input_box.clear()
        input_box.send_keys(japanese_text)

        try:
            pronunciation_elem = WebDriverWait(driver, 5).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div#sourceEditArea p span"))
            )
            pronunciation = pronunciation_elem.text
        except:
            pronunciation = "[발음 없음]"

        try:
            translation_elem = WebDriverWait(driver, 5).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div#txtTarget span"))
            )
            translation = translation_elem.text
        except:
            translation = "[번역 없음]"

    except Exception as e:
        print("❌ 오류 발생:", e)
        pronunciation = "[발음 없음]"
        translation = "[번역 없음]"

    return pronunciation, translation

def main():
    # ▶️ 경로 설정
    input_path = "../data/raw/raw_japanese_corpus.csv"
    result_path = "../data/processed/convert_japanese.csv"

    # ▶️ 원본 CSV 전체 불러오기
    df = pd.read_csv(input_path, header=None)

    if os.path.exists(result_path):
        existing = pd.read_csv(result_path)
        done_ids = set(existing["id"].astype(int))
        first_write = False
    else:
        done_ids = set()
        first_write = True

    remaining_rows = df[~df[0].isin(done_ids)]

    for _, row in remaining_rows.iterrows():
        sentence_id = row[0]
        jpn_text = str(row[2]).strip()

        pron, trans = get_papago_outputs(jpn_text)
        print(f"[{sentence_id}] {jpn_text} -> {pron} | {trans}")

        result_row = pd.DataFrame(
            [[sentence_id, jpn_text, pron, trans]],
            columns=["id", "japanese", "pronunciation", "translation"]
        )

        result_row.to_csv(
            result_path,
            mode="a",
            header=first_write,
            index=False,
            encoding="utf-8-sig"
        )
        first_write = False
        time.sleep(1.5)

    driver.quit()

if __name__ == "__main__":
    raise RuntimeError(
        "이 스크립트는 데이터 구축 과정을 설명하기 위한 코드입니다.\n"
        "실제 실행은 권장하지 않습니다."
    )