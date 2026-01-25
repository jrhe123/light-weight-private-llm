import re

content=""
with open("./华盖集.txt", "r", encoding="utf-8") as f:
    content = f.read()

lines = content.splitlines()
clean_lines = []
skip_mode = False

for line in lines:
    if re.match(r'^.*※.*※.*※.*$', line.strip()):
        skip_mode = True
        continue
    if skip_mode and re.match("^##.*$", line.strip()):
        skip_mode = False
        continue
    if not skip_mode:
        clean_lines.append(line)

clean_content = "\n".join(clean_lines)

#去掉 〔20〕
clean_content = re.sub(r'〔\d+〕', '', clean_content, flags=re.MULTILINE)

#1. 清除每行汉字不超过10个的行，包括（汉字，字符...)
clean_content = re.sub(r'^.{1,10}\n?$', '', clean_content, flags=re.MULTILINE)

#2. 清洗数据中所有的空格和空行
clean_content = re.sub(r'^\s+','', clean_content, flags=re.MULTILINE)

with open("./clean_data.txt", "w", encoding="utf-8") as f:
    f.write(clean_content)