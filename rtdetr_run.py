from ultralytics.models import RTDETR
if __name__ == '__main__':
    model = RTDETR(model='ultralytics/cfg/models/rt-detr/rtdetr-l.yaml')
    model.load('rtdetr-l.pt') # 不使用预训练权重可注释掉此行
    model.train(pretrained=True, data='own_datas_UEC/UEC.yaml', epochs=10, batch=16, device='cuda', imgsz=320,cache=False,)
