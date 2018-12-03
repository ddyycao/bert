import requests
import json

text = """泰国小众酒店！趁还没有人知道赶紧去拍照！
初来以为自己在希腊，关键词：性冷淡风 / 大热网红酒店 / 古迹景观 / 设计师作品各种标签，拍照凹造型想怎么拍就怎么拍。酒店是泰国著名设计师arisara chaktranon 的作品，环境非常安静，不过距离曼谷市区有点远（@猪哥爱自驾 有说过泰国自驾怎么玩）。非常适合闺蜜或者情侣去，附近有大城遗迹可以逛！
酒店名字：大城 sala ayutthaya 萨拉艾尤塔雅酒店
价格：700元左右/晚
地址：9/2 moo 4, u-thong road, pratu chai
phra nakhon si ayutthaya, 13000 thailand
交通：自驾最方便，推荐租车平台租租车(更多自驾攻略可戳@猪哥爱自驾   ）
预定方式：booking agoda等
我的小众旅行攻略  ins风  设计感超强的民宿  网红酒店  泰国旅行  这个地方超适合拍照  拍照圣地  东南亚旅行攻略  #曼谷酒店   曼谷旅行"""

r = requests.post('http://localhost:5000/poi', json={'input': text})

print(r.json())
