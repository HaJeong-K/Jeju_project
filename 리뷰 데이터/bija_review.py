from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import time
import pandas


driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()))  
URL = 'https://www.tripadvisor.co.kr/Attraction_Review-g297885-d2202426-Reviews-Bijarim_Forest-Jeju_Jeju_Island.html'
driver.get(url=URL)

korean_reviews_1 = []


try:
  
    review_elements = driver.find_elements(By.XPATH, '//*[@id="tab-data-qa-reviews-0"]/div/div[5]/div/div/div/div/div/div/div/span/span')
    for element in review_elements:
        korean_reviews_1.append(element.text)

except Exception as e:
    print(f"오류 발생: {str(e)}")

for idx, review in enumerate(korean_reviews_1):
    print(f"한글 리뷰 {idx + 1}: {review}")

korean_reviews_2 =[]

URL2 = 'https://www.tripadvisor.co.kr/Attraction_Review-g297885-d2202426-Reviews-or{}-Bijarim_Forest-Jeju_Jeju_Island.html'
for page_number in range(0, 101, 10):
    url = URL2.format(page_number)
    driver.get(url)
    
    try:
  
        review_elements = driver.find_elements(By.XPATH, '//*[@id="tab-data-qa-reviews-0"]/div/div[5]/div/div/div/div/div/div/div/span/span')
        for element in review_elements:
            korean_reviews_2.append(element.text)

    except Exception as e:
        print(f"오류 발생: {str(e)}")

    for idx, review in enumerate(korean_reviews_2):
        print(f"한글 리뷰 {idx + 1}: {review}")

udo_review=korean_reviews_1+korean_reviews_2

file_path = "bija_review.txt"
with open(file_path, "w", encoding="utf-8") as file:
    for item in udo_review:
        file.write(item+"\n")
