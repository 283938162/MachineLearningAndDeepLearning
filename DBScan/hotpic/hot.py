import csv
import pandas as pd

from urllib.request import urlopen, quote
import json

filepath = r'test.csv'

# 使用panda打开 能保持csv的原始格式
data = pd.read_csv(filepath)
print(data)


def getlnglat(address):
	"""根据传入地名参数获取经纬度"""
	url = 'http://api.map.baidu.com/geocoder/v2/'
	output = 'json'
	ak = 'Ti4YbLTnbrA32UvN3QFt20fK79ufDmds'  # 浏览器端密钥
	address = quote(address)
	uri = url + '?' + 'address=' + address + '&output=' + output + '&ak=' + ak
	req = urlopen(uri)
	res = req.read().decode()
	temp = json.loads(res)
	lat = temp['result']['location']['lat']
	lng = temp['result']['location']['lng']
	return lat, lng


file = open(r'经纬度.json', 'w')  # 建立json数据文件

with open(filepath, 'r', encoding = 'UTF-8') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		# print('line = ', line)
		if reader.line_num == 1:
			continue
		cellname = line[0].strip()
		price = line[1].strip()

		try:
			lat, lng = getlnglat(cellname)
			str_temp = '{"lat:"' + str(lat) + ',"lng":' + str(lng) + ',"count":' + str(price) + '},'
			print('str_temp = ', str_temp)
			file.write(str_temp)
			print("文件保存成功！")
		except:
			f = open("异常日志.txt", 'a')
			f.flush()
			f.close()
			print('发生异常')
file.close()
