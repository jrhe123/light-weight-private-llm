import requests
from bs4 import BeautifulSoup
import time
import os

#url
url = "https://www.mingzhuxiaoshuo.com/jinxiandai/111/"
base_url = "https://www.mingzhuxiaoshuo.com/"

#设置请求头，用于模拟浏览器访问
headers = {
    'User-Agent': 'Chrome'
}

print("爬虫开始工作......")

#获取主页页面
response = requests.get(url, headers=headers)
response.encoding = 'gbk'

if(response.status_code != 200):
    print("错误：无法获取主页{url}, 状态码：{response.status_code}")
    exit()

soup = BeautifulSoup(response.text, 'html.parser')

title = soup.find('h1').text.strip()
output_filename = f"{title}.txt"
print(f"爬取的数据集的名子是：{output_filename}")

#提取所有章节的链接和标题
chapter_list = soup.find('div', class_='list')

if not chapter_list:
    print("错误：在主页上未找到class='list' 的章节列表")
    exit()

chapter_tags = chapter_list.find_all('a')

chapter_infos = []
for tag in chapter_tags:
    chapter_url = base_url + tag['href']
    chapter_title = tag.text.strip()
    chapter_infos.append({'title':chapter_title, 'url':chapter_url})

print(f"共找到{len(chapter_infos)}个章节！")

#找开一个文件，用于保存每个章节的内容
with open(output_filename, 'w', encoding='utf-8') as f:
    for i, chapter in enumerate(chapter_infos):
        t = chapter['title']
        u = chapter['url']
        print(f"正在爬取{i+1}/{len(chapter_infos)}章:{t}, url:{u}")

        chapter_response = requests.get(u, headers=headers, timeout=10)
        chapter_response.encoding = 'gbk'

        if chapter_response.status_code == 200:
            chapter_soup = BeautifulSoup(chapter_response.text, 'html.parser')
            content_div = chapter_soup.find('div', id='content')

            if content_div:
                content_text = content_div.get_text(separator='\n').strip()
                f.write(f"##{t}\n\n")
                f.write(content_text)
                f.write("\n\n\n")
            else:
                print(f"警告：在页面{u}未找到正文内容！")
        else:
            print(f"警告：无法访问章节{u}, 状态码：{chapter_response.status_code}")

        time.sleep(1)