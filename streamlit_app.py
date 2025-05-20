import os
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"

from PIL import Image
import torch

import streamlit as st
import numpy as np
import cv2
from ultralytics import YOLO
import pandas as pd
from streamlit_drawable_canvas import st_canvas

stroke_width = 1
    
def rects2df(rects):
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
    sx, sy = st.session_state.sx, st.session_state.sy
    df['x'] /= sx
    df['y'] /= sy
    df['w'] /= sx
    df['h'] /= sy
    return df
    
def df2json(df):
    objects = []
    sx, sy = st.session_state.sx, st.session_state.sy
    for i in range(len(df)):
        t = df.iloc[i]
        ang = t['r']/180*np.pi
        objects.append({'left': sx*t['x'] - np.cos(ang)*(sx*t['w'] + stroke_width)/2 + np.sin(ang)*(sy*t['h'] + stroke_width)/2,
                        'top':  sy*t['y'] - np.sin(ang)*(sy*t['w'] + stroke_width)/2 - np.cos(ang)*(sy*t['h'] + stroke_width)/2,
                        'width': sx*t['w'],
                        'height': sy*t['h'],
                        'angle': t['r'],
                        'type': 'rect',
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
                        'ry': 0})
    return {'version': '4.4.0', 'objects': objects, 'background': ''}        
 
def infer_on_click():
    st.session_state.transcribe = 2
    
def clear_on_click():
    if not st.session_state.json_data:
        st.session_state.json_data = df2json(pd.DataFrame())
    else:
        st.session_state.json_data = None

def df_on_change():
    for i,d in st.session_state.df['edited_rows'].items():
        for k,v in d.items():
            st.session_state.result.loc[st.session_state.result.index[i],k] = v
    st.session_state.json_data = df2json(st.session_state.result)
        
def main():
    st.set_page_config(layout="wide")
    
    authors = ["Корф","Сухово-Кобылин", "Литке"]
    tasks = ["Прямые рамки", "Повёрнутые рамки", "Прямые рамки с маской"]
    model_names = {author: {task: None for task in tasks} for author in authors}
    model_names["Корф"]["Повёрнутые рамки"] = "models/korf-obb.pt"
    model_names["Сухово-Кобылин"]["Прямые рамки"] = "models/skob.pt"
    model_names["Сухово-Кобылин"]["Повёрнутые рамки"] = "models/skob-obb.pt"
    
    st.markdown("## Распознавание рукописного текста")
    st.sidebar.title("Параметры")

    author = st.sidebar.radio("Почерк: ", authors)
    task = st.sidebar.radio("Тип модели: ", tasks)
    
    imgh = st.sidebar.slider("Высота изображения", min_value=320, max_value=3200, value=1600, step=32)
    conf = st.sidebar.slider("Уверенность", min_value=0.1, max_value=1.0, value=0.5)
    iou = st.sidebar.slider("Доля пересечения", min_value=0.1, max_value=1.0, value=0.7)
    
    imfile = st.sidebar.file_uploader("Выберите изображение", type=["jpg","png","bmp"])
    col1,col2 = st.columns([0.5,0.5])
    
    if imfile:        
        if "image_id" not in st.session_state or imfile.file_id != st.session_state.image_id:
            img = cv2.imdecode(np.frombuffer(imfile.getvalue(), dtype='uint8'), cv2.IMREAD_COLOR)[:,:,(2,1,0)]
            st.session_state.image_id= imfile.file_id
            scale = 500/img.shape[1]
            canvas = cv2.resize(img, (round(img.shape[1]*scale), round(img.shape[0]*scale)))
            st.session_state.img = img
            st.session_state.canvas = Image.fromarray(canvas)
            st.session_state.sx = canvas.shape[1] / img.shape[1]
            st.session_state.sy = canvas.shape[0] / img.shape[0]
            st.session_state.json_data = None
            st.session_state.result = pd.DataFrame({'Расшифровка': [], 'x': [], 'y': [], 'w': [], 'h': [], 'r': []})
            
        if st.session_state.transcribe == 2:
            with col2:
                model_name = model_names[author][task]
                if model_name:
                    if "model_name" not in st.session_state or model_name != st.session_state.model_name:
                        st.session_state.model_name = model_name
                        st.session_state.model = YOLO(model_name)
                        print(f"Loading model {model_name}")
                    
                    if st.session_state.result.empty:
                        preds = None
                    else:
                        df = st.session_state.result
                        x,y,w,h,r = df['x'].to_numpy(),df['y'].to_numpy(),df['w'].to_numpy(),df['h'].to_numpy(),df['r'].to_numpy()
                        if st.session_state.model.task == 'detect':
                            preds = [torch.tensor(np.vstack([x-w/2, y-h/2, x+w/2, y+h/2, np.ones_like(x), np.zeros_like(x)]).transpose())]
                        elif st.session_state.model.task == 'obb':
                            preds = [torch.tensor(np.vstack([x, y, w, h, np.ones_like(x), np.zeros_like(x), r/180*np.pi]).transpose())]
                            
                    imgsz = 32 * round(imgh / st.session_state.img.shape[0] * max(st.session_state.img.shape[:2]) / 32) 
                    res = st.session_state.model.predict(st.session_state.img[:,:,(2,1,0)], imgsz=imgsz, conf=conf, iou=iou, preds=preds)[0]
                    if st.session_state.model.task == 'detect':
                        boxes = res.boxes.xywh.detach().cpu().numpy().astype('float64')
                        df = pd.DataFrame(boxes, columns=['x','y','w','h'])
                        df['Расшифровка'] = res.lines
                        df['r'] = 0.0
                    elif st.session_state.model.task == 'obb':
                        boxes = res.obb.xywhr.detach().cpu().numpy().astype('float64')
                        df = pd.DataFrame(boxes, columns=['x','y','w','h','r'])
                        df['Расшифровка'] = res.lines
                        df['r'] *= (180/np.pi)
                        df.loc[df['r'] >= 180, 'r'] -= 180
                        df.loc[df['r'] >= 90, 'r'] -= 180
                    
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
                    df = df[['Расшифровка','x','y','w','h','r']].set_index(np.arange(len(df))+1)
                    st.session_state.json_data = df2json(df)
                    st.data_editor(df, height=35*15+45, on_change=df_on_change, key='df')
                else:
                    st.data_editor(st.session_state.result, height=35*15+45, on_change=df_on_change, key='df')
                st.button('Расшифровать', key='infer_btn', on_click=infer_on_click)

        with col1:
            data = st_canvas(background_image=st.session_state.canvas,
                             width=st.session_state.canvas.size[0],
                             height=st.session_state.canvas.size[1],
                             stroke_color='red',
                             stroke_width=1,
                             fill_color="rgba(255, 255, 255, 0)",
                             drawing_mode='transform' if 'cbox' in st.session_state and st.session_state.cbox else 'rect',
                             initial_drawing=st.session_state.json_data,
                             key=imfile.name)
            col11,col12 = st.columns([0.5,0.5])
            with col11:
                st.checkbox("Режим корректировки рамок", value=False, key='cbox')
            with col12:
                st.button('Очистить', key='clear_btn', on_click=clear_on_click)
            if data.json_data is None:
                st.session_state.result = pd.DataFrame({'Расшифровка': [], 'x': [], 'y': [], 'w': [], 'h': [], 'a': []})
            else:
                st.session_state.result = rects2df(data.json_data['objects'])  
            
        if st.session_state.transcribe != 2:
            with col2:
                st.data_editor(st.session_state.result, height=35*15+45, on_change=df_on_change, key='df')
                st.button('Расшифровать', key='infer', on_click=infer_on_click)
        
        if st.session_state.transcribe > 0:
            with col2:
                if model_names[author][task]:
                    st.success('Страница успешно распознана')
                else:
                    st.warning('Модель данного типа ещё не подключена')
                    st.session_state.transcribe -= 1
    
    st.session_state.transcribe = 0 if 'transcribe' not in st.session_state else max(st.session_state.transcribe-1,0)
    
if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass