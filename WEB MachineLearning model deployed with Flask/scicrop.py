# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 13:07:55 2019

@author: William
"""
# scricrop.py - contains functions to run model
    

crop_list = [['Apple', 0],
 ['Arcanut (Processed)', 1],
 ['Arecanut', 2],
 ['Arhar/Tur', 3],
 ['Ash Gourd', 4],
 ['Atcanut (Raw)', 5],
 ['Bajra', 6],
 ['Banana', 7],
 ['Barley', 8],
 ['Bean', 9],
 ['Beans & Mutter(Vegetable)', 10],
 ['Beet Root', 11],
 ['Ber', 12],
 ['Bhindi', 13],
 ['Bitter Gourd', 14],
 ['Black pepper', 15],
 ['Blackgram', 16],
 ['Bottle Gourd', 17],
 ['Brinjal', 18],
 ['Cabbage', 19],
 ['Cardamom', 20],
 ['Carrot', 21],
 ['Cashewnut', 22],
 ['Cashewnut Processed', 23],
 ['Cashewnut Raw', 24],
 ['Castor seed', 25],
 ['Cauliflower', 26],
 ['Citrus Fruit', 27],
 ['Coconut ', 28],
 ['Coffee', 29],
 ['Colocosia', 30],
 ['Cond-spcs other', 31],
 ['Coriander', 32],
 ['Cotton(lint)', 33],
 ['Cowpea(Lobia)', 34],
 ['Cucumber', 35],
 ['Drum Stick', 36],
 ['Dry chillies', 37],
 ['Dry ginger', 38],
 ['Garlic', 39],
 ['Ginger', 40],
 ['Gram', 41],
 ['Grapes', 42],
 ['Groundnut', 43],
 ['Guar seed', 44],
 ['Horse-gram', 45],
 ['Jack Fruit', 46],
 ['Jobster', 47],
 ['Jowar', 48],
 ['Jute', 49],
 ['Jute & mesta', 50],
 ['Kapas', 51],
 ['Khesari', 52],
 ['Korra', 53],
 ['Lab-Lab', 54],
 ['Lemon', 55],
 ['Lentil', 56],
 ['Linseed', 57],
 ['Litchi', 58],
 ['Maize', 59],
 ['Mango', 60],
 ['Masoor', 61],
 ['Mesta', 62],
 ['Moong(Green Gram)', 63],
 ['Moth', 64],
 ['Niger seed', 65],
 ['Oilseeds total', 66],
 ['Onion', 67],
 ['Orange', 68],
 ['Other  Rabi pulses', 69],
 ['Other Cereals & Millets', 70],
 ['Other Citrus Fruit', 71],
 ['Other Dry Fruit', 72],
 ['Other Fresh Fruits', 73],
 ['Other Kharif pulses', 74],
 ['Other Vegetables', 75],
 ['Paddy', 76],
 ['Papaya', 77],
 ['Peach', 78],
 ['Pear', 79],
 ['Peas  (vegetable)', 80],
 ['Peas & beans (Pulses)', 81],
 ['Perilla', 82],
 ['Pineapple', 83],
 ['Plums', 84],
 ['Pome Fruit', 85],
 ['Pome Granet', 86],
 ['Potato', 87],
 ['Pulses total', 88],
 ['Pump Kin', 89],
 ['Ragi', 90],
 ['Rajmash Kholar', 91],
 ['Rapeseed &Mustard', 92],
 ['Redish', 93],
 ['Ribed Guard', 94],
 ['Rice', 95],
 ['Ricebean (nagadal)', 96],
 ['Rubber', 97],
 ['Safflower', 98],
 ['Samai', 99],
 ['Sannhamp', 100],
 ['Sapota', 101],
 ['Sesamum', 102],
 ['Small millets', 103],
 ['Snak Guard', 104],
 ['Soyabean', 105],
 ['Sugarcane', 106],
 ['Sunflower', 107],
 ['Sweet potato', 108],
 ['Tapioca', 109],
 ['Tea', 110],
 ['Tobacco', 111],
 ['Tomato', 112],
 ['Total foodgrain', 113],
 ['Turmeric', 114],
 ['Turnip', 115],
 ['Urad', 116],
 ['Varagu', 117],
 ['Water Melon', 118],
 ['Wheat', 119],
 ['Yam', 120],
 ['other fibres', 121],
 ['other misc. pulses', 122],
 ['other oilseeds', 123]]
def cropIntToName(num):
    '''
    Recebo um Inteiro. esse inteiro corresponde a um valor de Nome do Crop!
    '''
    return crop_list[num][0]






























