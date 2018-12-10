import requests
import json
import re





text = """🌠大阪景点 | 乘坐一次周身红色的Hep Five摩天轮，体验一次专属于大阪的浪漫💘

🎀情侣结伴来日本大阪旅游，坐摩天轮绝对是一个不可错过的体验！今天推荐给大家的这个矗立在Hep five商场顶部周身火红的摩天轮，它已经成了梅田甚至整个大阪的地标，相当耀眼。摩天轮行至最高点的时候可以俯瞰整个大阪的全景，不论是白天还是晚上都非常好看，要抓紧时间按下快门📸哦

🎆摩天轮不算很大，但漆成了饱和的红色，看起来很是可爱与甜蜜，晴天的时候，被白色建筑物和蓝天衬托出的红色摩天轮的吊篮非常优美，车厢内还有音箱，可以连接iphone播放音乐增添浪漫感。到了晚上，则周身环绕着红色光辉，成为大阪夜空中最亮的“星”

🎡景点名： Hep five摩天轮 (xxx)

🎡景点地址：Carrera de Mallorca 401，Barcelona

🎡门票价格：门票：500日元，小学生以下免费；凭大阪周游卡可免费乘坐一次！（划重点）

PS：大阪周游卡可以在租租车旅游商城上面买到哦

🎡开放时间：11:00-22:00

🎡交通方式：地下鉄御堂筋线梅田站下（可以用探途离线地图导航过去哦 ）

😉关注Nemo带你一分钟种草日韩美到爆的景点~喜欢记得点赞+收藏哦！"""


r = requests.post('http://localhost:5000/poi', json={'input': text})

print(r.json())
