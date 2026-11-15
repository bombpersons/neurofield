# All of MeshNCA's demo models as safetensors (NAME.w1: 81 x 128, inputs then bias; NAME.w2: 129 x
# 16), and a sheet of some grown on an 8-neighbour grid, for picking.
# meshnca_export.py MODELS_JSON OUT.safetensors SHEET.png NAME...
#   MODELS_JSON: data/models.json of github.com/meshnca/meshnca.github.io (no licence stated:
#   for testing only). OUT: runs/nca/meshnca_demo.safetensors for --dream-nca.
import sys, json, numpy as np, torch
from PIL import Image, ImageDraw
from safetensors.torch import save_file
src, OUT, SHEET, PICK = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
torch.set_num_threads(4)
m = json.load(open(src))
names = [n.split("Image_new_init_")[1].split("_sphere")[0] for n in m["model_names"]]
def unpack(layer, t, n_in, n_out):
    d = np.array(layer["data_flatten"], dtype=np.float32).reshape(layer["data_shape"])
    lx, ly = layer["layout"]; w4 = n_out // 4
    x0, y0 = (t % lx) * w4, (t // lx) * (n_in + 1)
    return (d[y0:y0 + n_in + 1, x0:x0 + w4].reshape(n_in + 1, n_out) - layer["center"]) * layer["scale"]
out = {}
for t, n in enumerate(names):
    out[f"{n}.w1"] = torch.tensor(unpack(m["layers"][0], t, 80, 128)).contiguous()
    out[f"{n}.w2"] = torch.tensor(unpack(m["layers"][1], t, 128, 16)).contiguous()
save_file(out, OUT)
print(len(names), "models")
DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
def perceive(s):
    nbs = [(torch.roll(s, (-dy, -dx), (2, 3)), (dx / np.hypot(dx, dy), -dy / np.hypot(dx, dy))) for dx, dy in DIRS]
    lap = sum(n - s for n, _ in nbs) * 0.5; k = 3 / (2 + 2 * np.sqrt(2))
    gx = sum(d[0] * (n - s) for n, d in nbs) * k; gy = sum(d[1] * (n - s) for n, d in nbs) * k
    return torch.cat([lap, gy, torch.zeros_like(s), gx, s], 1)
tiles = []
for n in PICK:
    W1, W2 = out[f"{n}.w1"], out[f"{n}.w2"]
    torch.manual_seed(0); s = torch.zeros(1, 16, 96, 96)
    with torch.no_grad():
        for _ in range(300):
            x = perceive(s).permute(0, 2, 3, 1)
            dx = (torch.relu(x @ W1[:80] + W1[80]) @ W2[:128] + W2[128]).permute(0, 3, 1, 2)
            s = s + dx * (torch.rand(1, 1, 96, 96) < 0.5)
    a = (s[0, :3] + 0.5).clamp(0, 1).permute(1, 2, 0).numpy(); h = ((s[0, 6] + 0.5).clamp(0, 1) ** 2).numpy()[..., None].repeat(3, -1)
    im = Image.fromarray((np.concatenate([a, h]) * 255).astype(np.uint8)).resize((192, 384), Image.NEAREST)
    ImageDraw.Draw(im).text((3, 3), n[:28], fill=(255, 255, 0)); tiles.append(im)
    print(n, flush=True)
sheet = Image.new("RGB", (192 * 6, 384 * ((len(tiles) + 5) // 6)))
for i, im in enumerate(tiles): sheet.paste(im, (i % 6 * 192, i // 6 * 384))
sheet.save(SHEET)
