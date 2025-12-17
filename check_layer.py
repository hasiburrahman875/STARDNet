from models.experimental import attempt_load
m = attempt_load("runs/train/mvaaod/stardnet_mvaaod/weights/last.pt", map_location="cpu")
for i, block in enumerate(m.model):
    print(i, block)
