import os
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"

from PIL import Image
from pillow_heif import register_heif_opener
register_heif_opener()
import io

import torch
import openai
from openai import OpenAI
import streamlit as st
import numpy as np
import cv2
from ultralytics import YOLO
import pandas as pd
from sklearn.cluster import KMeans
from streamlit_drawable_canvas import st_canvas
from ultralytics.utils.ops import xywhr2xyxyxyxy

stroke_width = 1
    
def rects2df(rects):
    sx, sy = st.session_state['sx'], st.session_state['sy']
    if not rects:
        df = pd.DataFrame({'Расшифровка': [], 'x': [], 'y': [], 'w': [], 'h': [], 'a': []})
    else:
        for i in range(len(rects)):
            if rects[i]['strokeLineCap'] == 'butt':
                rects[i]['strokeLineCap'] = ''
        df = pd.DataFrame(rects)
        w = df['width']*df['scaleX']
        h = df['height']*df['scaleY']
        ang = df['angle']/180*np.pi
        df.loc[df['angle'] >= 180, 'angle'] -= 180
        df.loc[df['angle'] >= 90, 'angle'] -= 180
        
        df = pd.DataFrame({'Расшифровка': df['strokeLineCap'],
                           'x': df['left'] + (w + stroke_width)*np.cos(ang)/2 - (h + stroke_width)*np.sin(ang)/2,
                           'y': df['top']  + (w + stroke_width)*np.sin(ang)/2 + (h + stroke_width)*np.cos(ang)/2,
                           'w': w,
                           'h': h,
                           'r': df['angle'].astype('float')
                          }).set_index(np.arange(len(df))+1)
        
        npts = max([len(r['path'])-1 for r in rects if r['type'] == 'path'] + [0])
        C = np.zeros((len(df), 2*npts))
        for i in range(len(rects)):
            if rects[i]['type'] == 'path':
                path = rects[i]['path'][:-1]
                x,y = [p[1] for p in path], [p[2] for p in path]
                C[i,0:2*len(path):2] = np.array(x) / sx
                C[i,1:2*len(path):2] = np.array(y) / sy
                df.loc[i+1,'x'] = (max(x) + min(x)) / 2
                df.loc[i+1,'y'] = (max(y) + min(y)) / 2
                
        df1 = pd.DataFrame(C, columns = sum([['x'+str(i+1), 'y'+str(i+1)] for i in range(npts)],[])).set_index(np.arange(len(df))+1)
        df = pd.concat([df,df1], axis=1)
    
    df['x'] /= sx
    df['y'] /= sy
    df['w'] /= sx
    df['h'] /= sy
    return df
    
def df2json(df):
    objects = []
    sx, sy = st.session_state['sx'], st.session_state['sy']
    
    pol = 'x1' in df.columns
    for i in range(len(df)):
        t = df.iloc[i]
        if pol:
            path = np.reshape(t[6:].values, (-1,2))
            path = path.tolist()
            path = [['L', sx*p[0], sy*p[1]] for p in path if p[0] != 0 or p[1] != 0]
            left,top = min(p[1] for p in path), min(p[2] for p in path)
            path[0][0] = 'M'
            path += ['z']
        else:
            path = []
        
        ang = t['r']/180*np.pi
        objects.append({'left':left if pol else sx*t['x']-np.cos(ang)*(sx*t['w'] + stroke_width)/2 + np.sin(ang)*(sy*t['h'] + stroke_width)/2,
                        'top': top  if pol else sy*t['y']-np.sin(ang)*(sy*t['w'] + stroke_width)/2 - np.cos(ang)*(sy*t['h'] + stroke_width)/2,
                        'width': sx*t['w'],
                        'height': sy*t['h'],
                        'angle': t['r'],
                        'type': 'path' if pol else 'rect',
                        'version': '4.4.0',
                        'originX': 'left',
                        'originY': 'top',
                        'fill': 'rgba(255, 255, 255, 0)',
                        'stroke': 'red',
                        'strokeWidth': 1,
                        'strokeDashArray': None,
                        'strokeLineCap': t['Расшифровка'],
                        'strokeDashOffset': 0,
                        'strokeLineJoin': 'miter',
                        'strokeUniform': True,
                        'strokeMiterLimit': 4,
                        'scaleX': 1,
                        'scaleY': 1,
                        'flipX': False,
                        'flipY': False,
                        'opacity': 1,
                        'shadow': None,
                        'visible': True,
                        'backgroundColor': '',
                        'fillRule': 'nonzero',
                        'paintFirst': 'fill',
                        'globalCompositeOperation': 'source-over',
                        'skewX': 0,
                        'skewY': 0,
                        'rx': 0,
                        'ry': 0,
                        'path': path})
    
    return {'version': '4.4.0', 'objects': objects, 'background': ''}        
 
def infer_on_click():
    model_name = st.session_state['model_names'][st.session_state['author']][st.session_state['task']]
    
    if model_name:
        if "model_name" not in st.session_state or model_name != st.session_state['model_name'] or 'model' not in st.session_state:
            st.session_state['model_name'] = model_name
            st.session_state['model'] = YOLO(model_name)
            print(f"Loading model {model_name}")
            if not hasattr(st.session_state['model'].model, 'ng'):
                st.session_state['model'].model.ng = 0
            
        imgsz = 32 * round(st.session_state['imgh'] / st.session_state['img'].shape[0] * max(st.session_state['img'].shape[:2]) / 32) 
        st.session_state['msg'] = ''

        if st.session_state['result'].empty:
            preds = None
        else:
            df = st.session_state['result']
            x,y,w,h,r = df['x'].to_numpy(),df['y'].to_numpy(),df['w'].to_numpy(),df['h'].to_numpy(),df['r'].to_numpy()
            if st.session_state['model'].task == 'detect':
                preds = [torch.tensor(np.vstack([x-w/2, y-h/2, x+w/2, y+h/2, np.ones_like(x), np.zeros_like(x)]).transpose())]
            elif st.session_state['model'].task == 'obb':
                preds = [torch.tensor(np.vstack([x, y, w, h, np.ones_like(x), np.zeros_like(x), r/180*np.pi]).transpose())]
            else:
                preds = np.vstack([np.zeros((4,len(x)),dtype=x.dtype), np.ones_like(x), np.zeros_like(x)]).transpose()
                masks = []
                for i in range(len(df)):
                    if all(df.loc[i+1].values[6:] == 0):
                        xywhr = df.loc[i+1].values[1:6].astype('float')
                        xywhr[-1] *= (np.pi/180)
                        xy = xywhr2xyxyxyxy(xywhr)
                        preds[i,:4] = [xy[:,0].min(), xy[:,1].min(), xy[:,0].max(), xy[:,1].max()]
                        masks.append(xy)
                    else:
                        masks.append(df.loc[i+1].values[6:].astype(float).reshape((-1,2)))
                        masks[-1] = masks[-1][np.any(masks[-1] != 0, axis=1)]
                preds = ([torch.tensor(preds)], [masks])
                    
        res = st.session_state['model'].predict(st.session_state['img'][:,:,(2,1,0)], imgsz=imgsz, conf=st.session_state['conf'], \
                                                iou=st.session_state['iou'], preds=preds)[0]
        mods = chr(818)+chr(821)+chr(819)
        res.lines = [''.join(chr(ord(c) % 10000) + ('' if ord(c) < 10000 else (chr(0)+mods)[ord(c) // 10000]) for c in line) \
                     for line in res.lines]
        
        wimg = st.session_state['img'].shape[1]
        dx = wimg*0.02
        wmin = wimg*0.10

        if st.session_state['model'].task == 'detect':
            boxes = res.boxes.xywh.detach().cpu().numpy().astype('float64')
            df = pd.DataFrame(boxes, columns=['x','y','w','h'])
            df['Расшифровка'] = res.lines
            df['r'] = 0.0
            mask = (res.boxes.xywh[:,2] > wmin) | ((res.boxes.xyxy[:,0] > dx) & (res.boxes.xyxy[:,2] < wimg - dx))
        elif st.session_state['model'].task == 'obb':
            boxes = res.obb.xywhr.detach().cpu().numpy().astype('float64')
            df = pd.DataFrame(boxes, columns=['x','y','w','h','r'])
            df['Расшифровка'] = res.lines
            df['r'] *= (180/np.pi)                        
            df.loc[df['r'] >= 180, 'r'] -= 180
            df.loc[df['r'] >= 90, 'r'] -= 180
            mask = (res.obb.xywhr[:,2] > wmin) | \
               ((res.obb.xyxyxyxy[:,:,0].amin(axis=1) > dx) & (res.obb.xyxyxyxy[:,:,0].amax(axis=1) < wimg - dx))
        elif st.session_state['model'].task == 'segment':
            boxes = res.boxes.xywh.detach().cpu().numpy().astype('float64')
            df = pd.DataFrame(boxes, columns=['x','y','w','h'])
            df['Расшифровка'] = res.lines
            df['r'] = 0.0
            
            npts = max([xy.shape[0] for xy in res.masks.xy])
            C = np.zeros((len(res.masks.xy), 2*npts))
            for i,xy in enumerate(res.masks.xy):
                C[i,0:2*xy.shape[0]:2] = xy[:,0]
                C[i,1:2*xy.shape[0]:2] = xy[:,1] 
            df1 = pd.DataFrame(C, columns=sum([['x' + str(i+1), 'y' + str(i+1)] for i in range(npts)],[]))
            df = pd.concat([df,df1], axis=1)

            boxes = res.boxes.xywh.detach().cpu().numpy().astype('float64')
            mask = (res.boxes.xywh[:,2] > wmin) | ((res.boxes.xyxy[:,0] > dx) & (res.boxes.xyxy[:,2] < wimg - dx))

        df = df[mask.detach().cpu().numpy()]
        df = sort_boxes(df)
        
        st.session_state['result'] = df
        st.session_state['init_draw'] = df2json(df)
        st.session_state['reinit'] = True
        st.session_state['df_key'] = 'df1' if st.session_state['df_key'] == 'df0' else 'df0'
        
        if st.session_state['gbox']:            
            try:
                content = st.session_state['content']
                prompt = "".join(l+'\n' for l in df['Расшифровка'])
                response = st.session_state['client'].chat.completions.create(
                    model="deepseek/deepseek-r1-0528:free",
                    messages=[{'role': 'system', 'content': content},
                              {'role': 'user', 'content': prompt}
                    ]
                )
                st.session_state['msg'] = response.choices[0].message.content
            except Exception as e:             
                st.session_state['msg'] = e.message
            except:
                st.session_state['msg'] = "Ошибка. Не удалось обработать запрос. Попробуйте повторить позже."
        else:
            st.session_state['msg'] = ''
    else:
        st.session_state['msg'] = "[Ошибка] Модель данного типа ещё не подключена."
    
def clear_on_click():
    st.session_state['init_draw'] = rects2df([]) if st.session_state['init_draw'] is None else None
    st.session_state['reinit'] = True
    
def cbox_on_change():
    st.session_state['init_draw'] = st.session_state[st.session_state['imfile'].name]['raw']
    st.session_state['reinit'] = True

def pbox_on_change():
    st.session_state['init_draw'] = st.session_state[st.session_state['imfile'].name]['raw']
    st.session_state['reinit'] = True
    
def sort_boxes(df):
    if len(df) < 2:
        return df
    lab = KMeans(n_clusters=2).fit(df['x'].values.reshape(-1,1)).labels_
    if df['x'][lab == 0].mean() > df['x'][lab == 1].mean():
        lab = 1 - lab
        
    l = df['x'][lab == 0].mean()
    r = df['x'][lab == 1].mean()
    w = df['w'].median()
    if r - l > w:
        dfl = df[lab == 0]
        dfr = df[lab == 1]
        dfl.sort_values(by='y', inplace=True)
        dfr.sort_values(by='y', inplace=True)
        df = pd.concat([dfl, dfr])
    else:
        df.sort_values(by='y', inplace=True)
    
    idx = np.arange(len(df))
    step = 10
    for i in range(len(idx)):
        for j in range(i+1,len(idx)):
            x1,y1,w1,h1 = df.iloc[idx[i]]['x'], df.iloc[idx[i]]['y'], df.iloc[idx[i]]['w'], df.iloc[idx[i]]['h']
            x2,y2,w2,h2 = df.iloc[idx[j]]['x'], df.iloc[idx[j]]['y'], df.iloc[idx[j]]['w'], df.iloc[idx[j]]['h']
            if x1-w1/2+step >= x2-w2/2 and x1+w1/2-step <= x2+w2/2 and y1-h1/2+step >= y2-h2/2 and y1+h1/2-step <= y2+h2/2:
                idx[[i,j]] = idx[[j,i]]

    df = df.reindex(index=df.index[idx])
    df = df[['Расшифровка','x','y','w','h','r'] + list(df.columns[6:])].set_index(np.arange(len(df))+1)
    
    return df

default_content = "Представь, что ты помогаешь историку-архивисту, работающему с рукописными документами 19-го века на русском языке в дореформенной орфографии. У него есть модель компьютерного зрения, распознающая рукописный текст. Так как модель несовершенна, в её расшифровке встречаются ошибки. В каждом из последующих сообщений будет полученная этой моделью расшифровка рукописной страницы, которую тебе нужно будет скорректировать. Твой ответ должен содержать только исправленный текст, переведённый в современную орфографию, без собственных комментариев."

def main():
    st.set_page_config(layout="wide")
    
    authors = ["Корф","Сухово-Кобылин", "Литке"]
    tasks = ["Прямые рамки", "Повёрнутые рамки", "Прямые рамки с маской"]
    
    st.markdown("## Распознавание рукописного текста")
    st.sidebar.title("Параметры")

    author = st.sidebar.radio("Почерк: ", authors, index=authors.index("Сухово-Кобылин"), key='author')
    task = st.sidebar.radio("Тип модели: ", tasks, index=tasks.index("Повёрнутые рамки"), key='task')
    
    st.sidebar.slider("Высота изображения", min_value=320, max_value=3200, value=1760, step=32, key='imgh')
    st.sidebar.slider("Уверенность", min_value=0.1, max_value=1.0, value=0.5, key='conf')
    st.sidebar.slider("Доля пересечения", min_value=0.1, max_value=1.0, value=0.7, key='iou')
    
    imfile = st.sidebar.file_uploader("Выберите изображение", type=["jpg","png","bmp","heic"])
    if imfile and ('imfile' not in st.session_state or imfile != st.session_state['imfile']):
        img = Image.open(io.BytesIO(imfile.getvalue()))
        exif = img.getexif()        
        img = np.asarray(img)
        if 274 in exif and exif[274] == 6:
            img = img.transpose(1,0,2)[:,::-1]
        scale = 500/img.shape[1]
        canvas = cv2.resize(img, (round(img.shape[1]*scale), round(img.shape[0]*scale)))
        st.session_state['img'] = img
        st.session_state['canvas'] = Image.fromarray(canvas)
        st.session_state['sx'] = canvas.shape[1] / img.shape[1]
        st.session_state['sy'] = canvas.shape[0] / img.shape[0]
        st.session_state['result'] = pd.DataFrame({'Расшифровка': [], 'x': [], 'y': [], 'w': [], 'h': [], 'r': []})
        st.session_state['msg'] = ''
        st.session_state['init_draw'] = None
        st.session_state['imfile'] = imfile
    
    col1,col2 = st.columns([0.5,0.5])
    
    if 'model_names' not in st.session_state:
        model_names = {author: {task: None for task in tasks} for author in authors}
        model_names["Корф"]["Прямые рамки"] = "./src/models/korf.pt"
        model_names["Корф"]["Повёрнутые рамки"] = "./src/models/korf-obb.pt"
        model_names["Корф"]["Прямые рамки с маской"] = "./src/models/korf-seg.pt"
        model_names["Сухово-Кобылин"]["Повёрнутые рамки"] = "./src/models/skob-obb.pt"
        model_names["Сухово-Кобылин"]["Прямые рамки"] = "./src/models/skob.pt"
        model_names["Литке"]["Прямые рамки"] = "./src/models/litke.pt"
        st.session_state['model_names'] = model_names
    if 'df_key' not in st.session_state:
        st.session_state['df_key'] = 'df0'
    if 'client' not in st.session_state:
        st.session_state['client'] = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv('OpenRouter_API_Key'),
        )
    
    if imfile:
        with col1:
            drawing_mode = 'transform' if 'cbox' in st.session_state and st.session_state['cbox'] else \
                           'polygon' if 'pbox' in st.session_state and st.session_state['pbox'] else \
                           'rect'
            data = st_canvas(background_image=st.session_state['canvas'],
                             update_streamlit=True,
                             width=st.session_state['canvas'].size[0],
                             height=st.session_state['canvas'].size[1],
                             stroke_color='red',
                             stroke_width=1,
                             fill_color="rgba(255, 255, 255, 0)",
                             drawing_mode=drawing_mode,
                             initial_drawing=st.session_state['init_draw'],
                             key=st.session_state['imfile'].name)
            
            if 'reinit' in st.session_state and st.session_state['reinit']:
                if st.session_state['init_draw'] is not None:
                    st.session_state['result'] = rects2df(st.session_state['init_draw']['objects'])
                else:
                    st.session_state['result'] = pd.DataFrame({'Расшифровка': [], 'x': [], 'y': [], 'w': [], 'h': [], 'a': []})
                st.session_state['reinit'] = False
            else:
                if data.json_data:
                    st.session_state['result'] = rects2df(data.json_data['objects'])
                else:
                    st.session_state['result'] = pd.DataFrame({'Расшифровка': [], 'x': [], 'y': [], 'w': [], 'h': [], 'a': []})
                        
            col1b,col2b,col3b = st.columns([0.39,0.39,0.22])
            with col1b:
                st.checkbox("Многоугольная рамка", value=False, key='pbox', on_change=pbox_on_change)
            with col2b:
                st.checkbox("Корректировка рамок", value=True, key='cbox', on_change=cbox_on_change)
            with col3b:
                st.button('Очистить', key='clear_btn', on_click=clear_on_click)
            
        with col2:
            st.data_editor(st.session_state['result'], height=35*20+45, key=st.session_state['df_key'])
            st.text_area('Запрос в DeepSeek', value=default_content, key='content')

            col1a,col2a = st.columns([0.3,0.7])
            with col1a:
                st.button('Расшифровать', key='infer_btn', on_click=infer_on_click)
            with col2a:
                st.checkbox("Исправить текст", value=False, key='gbox')

            if st.session_state['msg']:
                if '[Ошибка]' in st.session_state['msg']:
                    st.error(st.session_state['msg'].replace('.','\.').replace('[Ошибка]',''))
                elif 'error' in st.session_state['msg']:
                    st.error(st.session_state['msg'].replace('.','\.'))
                else:
                    st.success(st.session_state['msg'].replace('.','\.'))
    
if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass