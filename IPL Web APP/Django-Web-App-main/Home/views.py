from django.shortcuts import render, HttpResponse

import pandas as pd
import sklearn
import joblib
import random
import string

import pyautogui
import time 




# Create your views here.
def index(request):
    # return HttpResponse("This is homepage")
    return render(request, 'index.html')

def about(request):
    return render(request, 'about.html')

def services(request):
    return HttpResponse("This is services page")

def contact(request):
    # return HttpResponse("This is contact page")
    return render(request, 'contact.html')

def requestPrediciton(request):
    reloadModal= joblib.load('./Modals/pklmodal.pkl')
    if request.method=='POST':
        temp={}
        temp['seplen']=request.POST.get('seplen')
        temp['sepwid']=request.POST.get('sepwid')
        temp['petlen']=request.POST.get('petlen')
        temp['petwid']=request.POST.get('petwid')
        

        testdata=pd.DataFrame({'x':temp}).transpose()

        finaltestdata=testdata[['seplen','sepwid','petlen','petwid']]

        scoreData=reloadModal.predict(finaltestdata)

        context={'scoreData':scoreData}
        print(temp)
        
         
        print (testdata)
        print(finaltestdata)
        print(temp)
        print(scoreData)

    return render(request, 'irisRec.html',context)


def irisRec(request):
    return render(request, 'irisRec.html')

def spambot(request):
    if request.method=='POST':
        temp1=request.POST.get('message1')
        print(temp1)
        n=0
        m=10
        print (m)
        time.sleep(5)
        while (n<=m):
            pyautogui.typewrite(temp1)
            pyautogui.press("enter")
            n=n+1
    return render(request, 'spambot.html')


def passGenRes(request):
    if request.method=='POST':
        temp={}
        temp['rangeInput']=request.POST.get('rangeInput')


        temp['1']=request.POST.get('1')
        temp['2']=request.POST.get('2')
        temp['3']=request.POST.get('3')
        temp['4']=request.POST.get('4')

        print(temp)


        character_one= string.ascii_uppercase 
        character_two= string.digits
        character_three= string.punctuation
        character_four=string.ascii_lowercase
        char=len(character_three)-1


        x=int(temp['rangeInput'])

        
        if temp['1']== 'on' and temp['2']== 'on' and temp['3']== 'on'and temp['4']=='on':
            password="".join(character_one[random.randint(0,25)]+character_two[random.randint(0,8)]+character_three[random.randint(0,char)]+character_four[random.randint(0,25)] for i in range(0,int(x/2)))


        elif temp['1']== 'on' and temp['2'] and temp['3']== 'on':
            password="".join(character_one[random.randint(0,25)]+character_two[random.randint(0,8)]+character_three[random.randint(0,char)] for i in range(0,int(x/2)))
        elif temp['1']== 'on' and temp['2']== 'on' and temp['4']=='on':
            password="".join(character_one[random.randint(0,25)]+character_two[random.randint(0,8)]+character_four[random.randint(0,25)] for i in range(0,int(x/2)))
        elif temp['2']== 'on' and temp['3']== 'on'and temp['4']:
            password="".join(character_two[random.randint(0,8)]+character_three[random.randint(0,char)]+character_four[random.randint(0,25)] for i in range(0,int(x/2)))



        elif temp['1']== 'on' and temp['2']== 'on':
            password="".join(character_one[random.randint(0,25)]+character_two[random.randint(0,8)] for i in range(0,int(x/2)))
        elif temp['1']== 'on' and temp['3']== 'on':
            password="".join(character_one[random.randint(0,25)]+character_three[random.randint(0,char)] for i in range(0,int(x/2)))
        elif temp['1']== 'on' and temp['4']== 'on':
            password="".join(character_one[random.randint(0,25)]+character_four[random.randint(0,25)] for i in range(0,int(x/2)))
        elif temp['2']== 'on'and temp['3']:
            password="".join(character_two[random.randint(0,8)]+character_three[random.randint(0,char)] for i in range(0,int(x/2)))
        elif temp['2']== 'on'and temp['4']:
            password="".join(character_two[random.randint(0,8)]+character_four[random.randint(0,25)] for i in range(0,int(x/2)))
        elif temp['3']== 'on'and temp['4']:
            password="".join(character_three[random.randint(0,char)]+character_four[random.randint(0,25)] for i in range(0,int(x/2)))


        elif temp['1']== 'on':
            password="".join(character_one[random.randint(0,25)] for i in range(0,int(x)))

        elif temp['2']== 'on':
            password="".join(character_two[random.randint(0,8)] for i in range(0,int(x)))

        elif temp['3']== 'on':
            password="".join(character_three[random.randint(0,char)] for i in range(0,int(x)))

        elif temp['4']== 'on':
            password="".join(character_four[random.randint(0,25)] for i in range(0,int(x)))
        else:
            password="Re-try"
        
        # password="".join(character_one[random.randint(0,25)]+character_two[random.randint(0,8)]+character_three[random.randint(0,char)]+character_four[random.randint(0,25)] for i in range(0,int(x/2)))
        # print("".join(password[0:x]))

        return_password="".join(password[0:x])
        result={'return_password': return_password}

        print(result)

    return render(request,'passwordGenerator.html', result)

def passGen(request):
    return render(request,'passwordGenerator.html')


