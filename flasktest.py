# coding=utf-8
# =============================================
# @Time      : 2022-05-05 11:47
# @Author    : DongWei1998
# @FileName  : flasktest.py
# @Software  : PyCharm
# =============================================
import json,time,requests

def translator():
    url = f"http://10.19.234.179:5557/api/v1/translator"
    demo_text ={
        'text':"第三条线路是西宁平安区平安驿景区化隆夏琼寺景区群科新区哟面产业园循化骆驼泉景区街子镇撒拉尔故里撒拉尔水镇。"
    }

    headers = {
        'Content-Type': 'application/json'
    }
    start = time.time()
    result = requests.post(url=url, json=demo_text,headers=headers)
    end = time.time()
    if result.status_code == 200:
        obj = json.loads(result.text)
        print(obj)
    else:
        print(result)
    print('Running time: %s Seconds' % (end - start))


translator()