import ipaddress, json, math, os, re, socket
from datetime import date, datetime
from urllib.parse import urlparse, urlunparse
from typing import Any
import requests
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader, select_autoescape

TOKEN_URL='https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token'
STATISTICS_URL='https://sh.dataspace.copernicus.eu/statistics/v1'
load_dotenv()
TIMEWEB_BASE_URL=os.getenv('TIMEWEB_BASE_URL','https://agent.timeweb.cloud')
MCHS_HOST='55.mchs.gov.ru'
EVALSCRIPT='''//VERSION=3
function setup(){return {input:[{bands:["B04","B08","SCL","dataMask"]}],output:[{id:"ndvi",bands:1,sampleType:"FLOAT32"},{id:"dataMask",bands:1}]};}
function evaluatePixel(s){const bad=[1,3,6,8,9,10,11];const d=s.B08+s.B04;const ok=s.dataMask===1&&!bad.includes(s.SCL)&&d!==0;return ok?{ndvi:[(s.B08-s.B04)/d],dataMask:[1]}:{ndvi:[0],dataMask:[0]};}'''

app=FastAPI(title='АгроСтрах — проверка страхового события')
app.mount('/static',StaticFiles(directory='static'),name='static')
env=Environment(loader=FileSystemLoader('templates'),autoescape=select_autoescape())
def render(**ctx:Any): return HTMLResponse(env.get_template('index.html').render(**ctx))

def get_token():
    cid=os.getenv('COPERNICUS_CLIENT_ID'); sec=os.getenv('COPERNICUS_CLIENT_SECRET')
    if not cid or not sec: raise RuntimeError('Не заданы COPERNICUS_CLIENT_ID / COPERNICUS_CLIENT_SECRET')
    r=requests.post(TOKEN_URL,data={'grant_type':'client_credentials','client_id':cid,'client_secret':sec},timeout=30); r.raise_for_status(); return r.json()['access_token']

def bbox(lat,lon,r):
    dy=r/111320; c=math.cos(math.radians(lat)); dx=r/(111320*c); return [lon-dx,lat-dy,lon+dx,lat+dy]

def period_ndvi(token,lat,lon,radius,start,end):
    resy=10/111320; resx=10/(111320*math.cos(math.radians(lat)))
    p={'input':{'bounds':{'bbox':bbox(lat,lon,radius),'properties':{'crs':'http://www.opengis.net/def/crs/EPSG/0/4326'}},'data':[{'type':'sentinel-2-l2a','dataFilter':{'mosaickingOrder':'leastCC','maxCloudCoverage':80}}]},'aggregation':{'timeRange':{'from':start+'T00:00:00Z','to':end+'T23:59:59Z'},'aggregationInterval':{'of':'P1D'},'evalscript':EVALSCRIPT,'resx':abs(resx),'resy':abs(resy)}}
    rr=requests.post(STATISTICS_URL,headers={'Authorization':f'Bearer {token}'},json=p,timeout=120); rr.raise_for_status(); vals=[]
    for x in rr.json().get('data',[]):
        try: s=x['outputs']['ndvi']['bands']['B0']['stats']
        except KeyError: continue
        n=s.get('sampleCount',0)-s.get('noDataCount',0); m=s.get('mean')
        if m is not None and n>0: vals.append((m,n,x.get('interval',{}).get('from','')[:10]))
    if not vals:return None,0
    total=sum(n for _,n,_ in vals); return sum(m*n for m,n,_ in vals)/total,len(vals)

def normalize_url(u):
    p=urlparse(u.strip()); host=(p.hostname or '').lower(); path=re.sub(r'/+$','',p.path or '/')
    return urlunparse((p.scheme.lower(),host,path,'',p.query,''))
def host(u): return (urlparse(u).hostname or '').lower()
def is_mchs(u): return host(u)==MCHS_HOST or host(u).endswith('.'+MCHS_HOST)

def safe_fetch(url):
    p=urlparse(url)
    if p.scheme not in ('http','https') or not p.hostname: raise ValueError('Разрешены только http/https ссылки')
    # basic SSRF protection
    for info in socket.getaddrinfo(p.hostname,443 if p.scheme=='https' else 80,type=socket.SOCK_STREAM):
        ip=ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved: raise ValueError('Локальные/приватные адреса запрещены')
    r=requests.get(url,headers={'User-Agent':'Mozilla/5.0 AgroInsuranceStudentProject/1.0'},timeout=20,allow_redirects=True)
    r.raise_for_status()
    if len(r.content)>5_000_000: raise ValueError('Документ больше 5 МБ')
    ct=r.headers.get('content-type','')
    if 'html' not in ct and 'text' not in ct: raise ValueError('Сейчас автоматическая проверка поддерживает HTML/текстовые страницы')
    soup=BeautifulSoup(r.text,'html.parser')
    for x in soup(['script','style','noscript']): x.decompose()
    text=' '.join(soup.get_text(' ',strip=True).split())
    return str(r.url), text[:60000]

def _extract_json(text):
    text=(text or '').strip()
    text=re.sub(r'^```(?:json)?\s*','',text,flags=re.I); text=re.sub(r'\s*```$','',text)
    a=text.find('{'); b=text.rfind('}')
    if a>=0 and b>a: text=text[a:b+1]
    return json.loads(text)

def deepseek_verify(url,text,kind,region,district,event_date,policy_start,policy_end):
    agent_id=os.getenv('TIMEWEB_AGENT_ACCESS_ID'); token=os.getenv('TIMEWEB_AGENT_TOKEN'); model=os.getenv('TIMEWEB_MODEL','').strip()
    diag={'api':'NOT_CALLED','http':None,'model':model or 'auto','page_chars':len(text),'semantic_relevance':None}
    if not agent_id or not token: return {'status':'UNVERIFIED','reason':'Не заданы TIMEWEB_AGENT_ACCESS_ID / TIMEWEB_AGENT_TOKEN в .env','facts':{},'diagnostics':diag}
    try:
        if not model:
            mr=requests.get(f'{TIMEWEB_BASE_URL}/api/v1/cloud-ai/agents/{agent_id}/v1/models',headers={'Authorization':f'Bearer {token}'},timeout=30); diag['http']=mr.status_code; mr.raise_for_status()
            models=mr.json().get('data',[])
            if not models:return {'status':'UNVERIFIED','reason':'Timeweb Agent API не вернул доступных моделей','facts':{},'diagnostics':diag}
            model=models[0]['id']; diag['model']=model
        prompt=f"""Ты проверяешь, является ли веб-страница разумным подтверждением засухи для учебного страхового прототипа. Анализируй ТОЛЬКО переданный текст страницы и ничего не выдумывай.
Источник: {kind}. URL: {url}
Контекст: регион={region}; район={district}; дата события={event_date}; полис={policy_start}..{policy_end}.
Проверка МЯГКАЯ и семантическая. Страница подходит, если из текста разумно следует, что в Омской области в 2026 году или примерно рядом с заявленным периодом была засуха, почвенная засуха, суховей, засушливые условия или ЧС из-за таких явлений.
Не требуй точной фразы «засуха». Если сказано «на территории Омской области», не требуй отдельного района. Не требуй точной даты начала/конца: дата публикации, распоряжения или сообщения в июле/августе 2026 подходит, если контекст явно про это событие. Не требуй официальный акт: новость/сообщение тоже подходит. Явно другой регион, другой год или несвязанная тема — отклони. Не решай PAY/NO_PAY и не проверяй срок полиса.
Верни ТОЛЬКО JSON:
{{"relevant":false,"confidence":"LOW","supports_drought_event":false,"region_matches":false,"time_relevant":false,"whole_region":false,"mentioned_district":null,"mentioned_date_or_period":null,"matched_terms":[],"evidence":"","explanation":""}}
confidence: HIGH, MEDIUM или LOW. relevant=true при HIGH/MEDIUM, если страница в целом разумно подтверждает событие. Не занижай из-за отсутствия точного района или даты начала.
ТЕКСТ СТРАНИЦЫ:\n{text}"""
        endpoint=f'{TIMEWEB_BASE_URL}/api/v1/cloud-ai/agents/{agent_id}/v1/chat/completions'
        rr=requests.post(endpoint,headers={'Authorization':f'Bearer {token}','Content-Type':'application/json'},json={'model':model,'messages':[{'role':'user','content':prompt}],'temperature':0},timeout=120); diag['http']=rr.status_code; rr.raise_for_status()
        facts=_extract_json(rr.json()['choices'][0]['message']['content']); confidence=str(facts.get('confidence','LOW')).upper()
        ok=bool(facts.get('relevant')) or (bool(facts.get('supports_drought_event')) and bool(facts.get('region_matches')) and confidence in ('HIGH','MEDIUM'))
        diag.update({'api':'OK','semantic_relevance':confidence})
        return {'status':'VERIFIED' if ok else 'UNVERIFIED','reason':None if ok else 'DeepSeek не нашёл достаточно релевантного подтверждения засухи в Омской области в рассматриваемом периоде','facts':facts,'diagnostics':diag}
    except requests.HTTPError as e:
        code=getattr(e.response,'status_code',None); diag.update({'api':'ERROR','http':code}); return {'status':'UNVERIFIED','reason':f'Timeweb/DeepSeek API: HTTP {code}','facts':{},'diagnostics':diag}
    except Exception as e:
        diag['api']='ERROR'; return {'status':'UNVERIFIED','reason':f'Ошибка анализа DeepSeek: {e}','facts':{},'diagnostics':diag}

def verify_source(url,kind,region,district,event_date,policy_start,policy_end):
    try:
        final,text=safe_fetch(url)
        result=deepseek_verify(final,text,kind,region,district,event_date,policy_start,policy_end); result['url']=final; result['host']=host(final); return result
    except Exception as e: return {'status':'UNVERIFIED','reason':str(e),'facts':{},'url':url,'host':host(url)}

@app.get('/',response_class=HTMLResponse)
def home(request:Request):
    return render(result=None,error=None,values={'region':'Омская область','radius':3000,'threshold':20,'advance_share':30,'policy_start':'2026-05-01','policy_end':'2026-09-30','event_date':'2026-07-25'})

@app.post('/analyze',response_class=HTMLResponse)
def analyze(request:Request, region:str=Form(...),district:str=Form(...),crop:str=Form(''),area_ha:float=Form(...),policy_start:str=Form(...),policy_end:str=Form(...),event_date:str=Form(...),insured_amount:float=Form(...),advance_share:float=Form(30),latitude:float=Form(...),longitude:float=Form(...),radius:float=Form(3000),threshold:float=Form(20),act_url:str=Form(...),independent_url:str=Form(...)):
    v=locals().copy(); v.pop('request',None)
    try:
        if normalize_url(act_url)==normalize_url(independent_url): raise ValueError('Источник 2 и источник 3 не могут быть одной и той же ссылкой.')
        if host(act_url)==host(independent_url): raise ValueError('Источники 2 и 3 должны быть независимыми: используйте разные домены/организации.')
        if not is_mchs(independent_url): raise ValueError('Независимый источник №3 должен быть с домена 55.mchs.gov.ru.')
        if is_mchs(act_url): raise ValueError('Источник №2 должен быть с другого домена, потому что источник №3 уже 55.mchs.gov.ru.')
        ps=date.fromisoformat(policy_start); pe=date.fromisoformat(policy_end); ed=date.fromisoformat(event_date)
        if ps>pe: raise ValueError('Начало полиса позже окончания.')
        if not (0<advance_share<=100): raise ValueError('Доля аванса должна быть 1–100%.')
        if not (-90<=latitude<=90 and -180<=longitude<=180): raise ValueError('Некорректные координаты.')
        token=get_token()
        periods=[('Конец июня 2025','2025-06-20','2025-06-30'),('Начало августа 2025','2025-08-01','2025-08-10'),('Конец июня 2026','2026-06-20','2026-06-30'),('Начало августа 2026','2026-08-01','2026-08-10')]
        nd=[]
        for label,a,b in periods:
            m,n=period_ndvi(token,latitude,longitude,radius,a,b); nd.append({'label':label,'value':m,'days':n})
        if any(x['value'] is None for x in nd): ndvi={'status':'UNVERIFIED','reason':'Недостаточно безоблачных Sentinel-2 данных','rows':nd}
        else:
            drop26=(nd[3]['value']-nd[2]['value'])/abs(nd[2]['value'])*100 if abs(nd[2]['value'])>.0001 else 0
            drop25=(nd[1]['value']-nd[0]['value'])/abs(nd[0]['value'])*100 if abs(nd[0]['value'])>.0001 else 0
            stress=drop26<=-abs(threshold) and drop26<drop25
            ndvi={'status':'VERIFIED' if stress else 'REJECTED','reason':None if stress else 'Порог NDVI-стресса не выполнен','rows':nd,'drop26':drop26,'drop25':drop25}
        act=verify_source(act_url,'general_drought_source',region,district,event_date,policy_start,policy_end)
        independent=verify_source(independent_url,'mchs_drought_source',region,district,event_date,policy_start,policy_end)
        # deterministic contract: AI never decides payment
        log=['POLICY_ACTIVE','EVENT_REPORTED','CHECKING']
        missing=[]; rejection=[]
        if not (ps<=ed<=pe): rejection.append('EVENT_OUTSIDE_POLICY_PERIOD')
        if ndvi['status']=='REJECTED': rejection.append('NDVI_CONDITION_NOT_MET')
        elif ndvi['status']!='VERIFIED': missing.append('NDVI')
        if act['status']!='VERIFIED': missing.append('SOURCE_2')
        if independent['status']!='VERIFIED': missing.append('SOURCE_3_MCHS')
        verified=sum(x=='VERIFIED' for x in [ndvi['status'],act['status'],independent['status']])
        if rejection: decision='NO_PAY'; reason=', '.join(rejection); state='REJECTED'
        elif missing: decision='NOT_ENOUGH_DATA'; reason='Не подтверждено: '+', '.join(missing); state='WAITING_DATA'
        elif verified<2: decision='NOT_ENOUGH_DATA'; reason='Нужно минимум 2 независимых verified-подтверждения'; state='WAITING_DATA'
        else: decision='PAY'; reason='Все условия подтверждены'; state='ADVANCE_PAID'
        log += [f'NDVI → {ndvi["status"]}',f'ACT → {act["status"]}',f'INDEPENDENT → {independent["status"]}',f'independent_verified={verified}',state]
        result={'decision':decision,'reason':reason,'state':state,'advance':insured_amount*advance_share/100 if decision=='PAY' else 0,'ndvi':ndvi,'act':act,'independent':independent,'log':log,'verified':verified,'policy_ok':ps<=ed<=pe}
        return render(result=result,error=None,values=v)
    except Exception as e: return render(result=None,error=str(e),values=v)

DEMO_SCENARIOS = {
    '1': {
        'title': 'Сценарий 1 · Выплата',
        'description': 'Полис действует, событие внутри периода, район покрыт актом, NDVI и два независимых источника подтверждены.',
        'decision': 'PAY', 'state': 'ADVANCE_PAID', 'reason': 'Все условия подтверждены',
        'advance': 1500000, 'verified': 3, 'policy_ok': True,
        'ndvi': {'status':'VERIFIED','reason':None,'rows':[
            {'label':'Конец июня 2025','value':0.61,'days':3},{'label':'Начало августа 2025','value':0.58,'days':3},
            {'label':'Конец июня 2026','value':0.63,'days':3},{'label':'Начало августа 2026','value':0.39,'days':4}], 'drop26':-38.1,'drop25':-4.9},
        'act': {'status':'VERIFIED','host':'publication.pravo.gov.ru','reason':None,'facts':{'evidence':'Акт ЧС: засуха, нужная территория и период подтверждены.'}},
        'independent': {'status':'VERIFIED','host':'55.mchs.gov.ru','reason':None,'facts':{'evidence':'Независимый источник МЧС подтверждает засуху в том же периоде.'}},
        'log':['POLICY_ACTIVE','EVENT_REPORTED','CHECKING','NDVI → VERIFIED','ACT → VERIFIED','INDEPENDENT → VERIFIED','independent_verified=3','RULE_ALL_CONDITIONS_MET','ADVANCE_PAID']
    },
    '2': {
        'title':'Сценарий 2 · Полис закончился','description':'Доказательства есть, но заявленная дата события позже даты окончания полиса.',
        'decision':'NO_PAY','state':'REJECTED','reason':'EVENT_OUTSIDE_POLICY_PERIOD','advance':0,'verified':3,'policy_ok':False,
        'ndvi': {'status':'VERIFIED','reason':None,'rows':[], 'drop26':-32.0,'drop25':-3.0},
        'act': {'status':'VERIFIED','host':'publication.pravo.gov.ru','reason':None,'facts':{'evidence':'Акт подтверждён.'}},
        'independent': {'status':'VERIFIED','host':'55.mchs.gov.ru','reason':None,'facts':{'evidence':'МЧС подтверждает событие.'}},
        'log':['POLICY_ACTIVE','EVENT_REPORTED','CHECKING','POLICY_PERIOD → FAILED','NDVI → VERIFIED','ACT → VERIFIED','INDEPENDENT → VERIFIED','RULE_EVENT_OUTSIDE_POLICY_PERIOD','REJECTED']
    },
    '3': {
        'title':'Сценарий 3 · Район не входит в акт','description':'Засуха подтверждена источниками, но территория хозяйства не входит в территорию действия акта ЧС.',
        'decision':'NO_PAY','state':'REJECTED','reason':'DISTRICT_NOT_COVERED','advance':0,'verified':2,'policy_ok':True,
        'ndvi': {'status':'VERIFIED','reason':None,'rows':[], 'drop26':-29.0,'drop25':-2.0},
        'act': {'status':'REJECTED','host':'publication.pravo.gov.ru','reason':'Акт найден, но указанный район не входит в территорию действия.','facts':{'evidence':'Территория акта не совпадает с районом хозяйства.'}},
        'independent': {'status':'VERIFIED','host':'55.mchs.gov.ru','reason':None,'facts':{'evidence':'МЧС подтверждает засуху, но это не расширяет территорию официального акта.'}},
        'log':['POLICY_ACTIVE','EVENT_REPORTED','CHECKING','NDVI → VERIFIED','ACT → REJECTED: DISTRICT_NOT_COVERED','INDEPENDENT → VERIFIED','RULE_DISTRICT_NOT_COVERED','REJECTED']
    },
    '4': {
        'title':'Сценарий 4 · Недостаточно данных','description':'Спутниковый признак подтверждён, но акт ЧС и независимый источник невозможно автоматически подтвердить.',
        'decision':'NOT_ENOUGH_DATA','state':'WAITING_DATA','reason':'Не подтверждено: EMERGENCY_ACT, INDEPENDENT_SOURCE','advance':0,'verified':1,'policy_ok':True,
        'ndvi': {'status':'VERIFIED','reason':None,'rows':[], 'drop26':-35.0,'drop25':-4.0},
        'act': {'status':'UNVERIFIED','host':'publication.pravo.gov.ru','reason':'Источник недоступен или документ невозможно проверить.','facts':{}},
        'independent': {'status':'UNVERIFIED','host':'55.mchs.gov.ru','reason':'Источник недоступен или публикация не подтверждена.','facts':{}},
        'log':['POLICY_ACTIVE','EVENT_REPORTED','CHECKING','NDVI → VERIFIED','ACT → UNVERIFIED','INDEPENDENT → UNVERIFIED','independent_verified=1','RULE_NOT_ENOUGH_DATA','WAITING_DATA']
    }
}

@app.get('/demo/{scenario_id}', response_class=HTMLResponse)
def demo(request: Request, scenario_id: str):
    scenario = DEMO_SCENARIOS.get(scenario_id)
    if not scenario:
        return render(result=None,error='Неизвестный демонстрационный сценарий',values={},demo=None)
    result=dict(scenario)
    return render(result=result,error=None,values={'region':'Омская область','radius':3000,'threshold':20,'advance_share':30,'policy_start':'2026-05-01','policy_end':'2026-09-30','event_date':'2026-07-25'},demo=scenario)
