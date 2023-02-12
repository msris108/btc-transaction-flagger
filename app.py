import requests
import json
import joblib
import pickle
import sklearn
import numpy as np
import pandas as pd
import xgboost as xgb
import streamlit as st

st.write("""
	# Bitcoin Transaction Classifier
	#### Enter Bitcoin Transaction Hash
""")


txhash = st.text_input('')

if txhash:
	# st.write(txn)
	data = requests.get("https://blockchain.info/rawtx/" + str(txhash))

	in_btc = 0
	out_btc = 0
	total_btc = 0

	dataJson = json.loads(data.content)
	st.write()

	if len(dataJson) > 2:

		indegree = dataJson['vin_sz']
		outdegree = dataJson['vout_sz']


		for i in dataJson["inputs"]:
			in_btc = in_btc + int(i["prev_out"]["value"])

		for i in dataJson["out"]:
			out_btc = out_btc + int(i["value"])

		in_btc = in_btc * 0.00000001

		out_btc = out_btc * 0.00000001

		total_btc = in_btc + out_btc

		mean_in_btc = in_btc/indegree

		mean_out_btc = out_btc/outdegree

		df = pd.DataFrame(np.array([[indegree, outdegree, in_btc, out_btc, total_btc]]), columns=['In Degree', 'Out Degree', 'In BTC', 'Out BTC', 'Total BTC'])

		# pca = joblib.load('pca.joblib')
		pca = pickle.load(open('pca.pkl', 'rb'))
		x = pca.transform(np.array([[indegree, outdegree, in_btc, out_btc, total_btc, mean_in_btc, mean_out_btc]]))
		# rfc = pickle.load(open('rfc.pkl', 'rb'))
		xgb = xgb.XGBClassifier()
		xgb.load_model('xgb.json')
		pred = xgb.predict(x)

		df.reset_index(drop=True, inplace=True)

		if pred == 0:
			st.write(df)
			st.write('### Safe Transaction ✅')
		else:
			st.write(df)
			st.write('### Illicit Transaction ❌')

	else:
		st.write('### ❌ Invalid Address')
		st.write(dataJson)




# Normal
# 0437cd7f8525ceed2324359c2d0ba26006d92d856a9c20fa0241106ee5a597c9
# f4184fc596403b9d638783cf57adfe4c75c605f6356fbc91338530e9831e9e16
# ea44e97271691990157559d0bdd9959e02790c34db6c006d779e82fa5aee708e

# ea44e97271691990157559d0bdd9959e02790c34db6c006d779e82fa5aee708e
# a16f3ce4dd5deb92d98ef5cf8afeaf0775ebca408f708b2146c4fb42b41e14be
# 591e91f809d716912ca1d4a9295e70c3e78bab077683f79350f101da64588073
# 298ca2045d174f8a158961806ffc4ef96fad02d71a6b84d9fa0491813a776160
# 12b5633bad1f9c167d523ad1aa1947b2732a865bf5414eab2f9e5ae5d5c191ba
# 4385fcf8b14497d0659adccfe06ae7e38e0b5dc95ff8a13d7c62035994a0cd79
# 828ef3b079f9c23829c56fe86e85b4a69d9e06e5b54ea597eef5fb3ffef509fe
# a3b0e9e7cddbbe78270fa4182a7675ff00b92872d8df7d14265a2b1e379a9d33
# 0cc917bf15f8807f224e7524c1eca22c3740ddefb7bf6694f7c2262b490cc706
# e8160a014fbff8386548f40205d540ef92ce8207ff4ac0446d6e591c6cf28f2c
# c3f0bb699bcc8a4e0716de45aef74c40aabeb80f7f00b3bdb45e115ee6f5400f
# 4d6edbeb62735d45ff1565385a8b0045f066055c9425e21540ea7a8060f08bf2
# 00ff9e64c9a2e7793e6f8c2b04072b4b22648cdedd46cd1c3ae3d6a23c8ec1eb
# d71fd2f64c0b34465b7518d240c00e83f6a5b10138a7079d1252858fe7e6b577
# e7caf9a784751643f7b71881aaf96e2b3e041950b42638b4fcfe82ff57ba260d
# 6bf363548b08aa8761e278be802a2d84b8e40daefe8150f9af7dd7b65a0de49f
# 6a71cea2c4e66ea163932b1ea199c1056f6728f3e1287946ed2a0892b918bf0e
# 59bf8acbc9d60dfae841abecc3882b4181f2bdd8ac6c1d94001165ab3aef50b0
# 04256336e9287f3b46508888cf3539dc0ab2fc8803cbe9668749fd18fc5dee85


# Illicit
# 52093f1f0c88f21966817bc6593fcf3e2cb0c314099182807057e7057ebfb2f7
# b19d1d08b5d186e3c1390aeb8d0341fb39cc2ccd4c90a1757f09d6fe756465d8
# d329685c70c218ef14c73cc936021dfae9a5de3ae737c3a15587ee593b3d1dde

# b19d1d08b5d186e3c1390aeb8d0341fb39cc2ccd4c90a1757f09d6fe756465d8
# eba46da607d544669ddd0262d908039e37d95f0a1fcd8b44b9cff66e6329a4a7
# 1aed1bca7d183f4f7b55112d88f9c0249cdbe4fd5f5b12e87da583e09cd8b92c
# 61aa58ea783ef5ff06d0c7c98de47f5555af0cf9f80ee53d2d477f6a80792720
# b135b3baa5eab9aca88ddffc328a4f062b88318c02b84b8bfd93d157ed0379f5
# b1582ce916529ef8525fef9756174f6905515a4ed071182cec1d072617316ece
# 8bc2c950e11d8e43bcdd3ec95ee7be959a0d8f6861bb94450c026cd1dcb00bba
# 8991eb3fb69480032a9425d38b5b7e6cd5ab129dbf098185c6235a5b9b0235f9
# 485265473de0c0fbfe5fee42e10ef70d0b5563aedede6da246cdbd2d761f7176
# 67a1895522a4a5a4240adff61c478326994204b308d27a5f958c4f953717fe12
# 83d2ea730fdc0a5fc21a7a4633b2cf417e3654c22f90396d72f8ae682f9de1a9
# 8facb5e8e0878daac1ffc44073d66d9b9e60c48d78535604b4edd6fa72b364f0
# 897ff16fcfa0c0d2a9d468a7d576753acfe26ec218bb5b19118f810460b8f2cf
# b3a91e53d6c1cbc14d3cf76ba3ed5e4764435e9f9973842dac18c5ac62438e93
# d28a40ea743d114f9e9aaedc40278589eceb943979dfa136490811552aa8a1f6
# 693fc4d3048de003349045a08958ae714785b52f6f7ad326127c937bb6c3f75f
