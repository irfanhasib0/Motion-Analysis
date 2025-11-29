import os, math
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

# ----------------------------
# 1) Model loading (DINO / DINOv2)
# ----------------------------
def load_dino(device="cuda"):
    """
    Tries to load a ViT-S/8 DINO (facebookresearch/dino).
    Falls back to DINOv2 (timm) if available.
    Returns: model, img_size, patch_stride, feat_dim, preprocess callable
    """
    # Option A: original DINO (ViT-S/8)
    try:
        model = torch.hub.load('facebookresearch/dino:main', 'dino_vits8')
        model.eval().to(device)
        img_size = 224
        patch_stride = 8   # ViT-S/8 -> 8x8 patches
        feat_dim = 384     # for ViT-S
        def preprocess(pil):
            pil = pil.resize((img_size, img_size), Image.BICUBIC)
            x = torch.from_numpy(np.array(pil)).float() / 255.0
            x = x.permute(2,0,1)  # HWC->CHW
            # DINO normalization
            mean = torch.tensor([0.485, 0.456, 0.406])[:,None,None]
            std  = torch.tensor([0.229, 0.224, 0.225])[:,None,None]
            x = (x - mean) / std
            return x
        backbone = "dino_vits8"
        return model, [img_size, img_size], patch_stride, feat_dim, preprocess, backbone
    except Exception:
        pass

    # Option B: DINOv2 via timm (e.g., vit_small_patch14_dinov2)
    try:
        import timm
        model = timm.create_model('vit_small_patch14_dinov2.lvd142m', pretrained=True)
        model.eval().to(device)
        img_size = 448         # 448 with 14px patch -> 32x32 tokens (nice grid)
        patch_stride = 14
        feat_dim = model.num_features
        data_cfg = timm.data.resolve_model_data_config(model)
        transforms = timm.data.create_transform(**data_cfg, is_training=False)
        def preprocess(pil):
            return transforms(pil)  # CHW tensor normalized as per model
        backbone = "dinov2_vit_s14"
        return model, img_size, patch_stride, feat_dim, preprocess, backbone
    except Exception as e:
        raise RuntimeError(
            "Could not load DINO/DINOv2. Install either:\n"
            " - torch.hub + facebookresearch/dino\n"
            " - timm==0.9+ with a DINOv2 variant\n"
            f"Original error: {e}"
        )

# ----------------------------
# 2) Feature extraction (patch tokens, not CLS)
# ----------------------------
@torch.no_grad()
def extract_patch_tokens(model, x, backbone):
    """
    x: (B,3,H,W) normalized tensor
    Returns feats as (B, C, Ht, Wt) where Ht,Wt are token grid size.
    """
    if backbone == "dino_vits8":
        # facebookresearch/dino ViT exposes get_last_selfattention etc.
        # But we want patch tokens: run forward and tap the token sequence.
        # The DINO ViT from torch.hub returns a VisionTransformer with .patch_embed and .forward() building tokens.
        # We’ll use a small hook to capture tokens pre-head.
        tokens = []

        def hook(module, inp, out):
            # out is token sequence (B, N+1, C); remove CLS
            seq = out[:, 1:, :]  # (B, N, C)
            B, N, C = seq.shape
            # Infer grid
            Ht = Wt = int(math.sqrt(N))
            tokens.append(seq.transpose(1,2).reshape(B, C, Ht, Wt))  # (B,C,Ht,Wt)

        h = model.norm.register_forward_hook(hook)
        _ = model(x)  # forward to trigger hook
        h.remove()
        feats = tokens[0]
        return F.normalize(feats, dim=1)

    elif backbone == "dinov2_vit_s14":
        # timm ViT: hook the pre-head features (model.forward_features)
        tokens = []
        def hook_feats(mod, inp, out):
            # out is dict or tensor depending on timm version
            if isinstance(out, dict) and 'x_norm_clstoken' in out:
                x = out['x_norm_patchtokens']  # (B, N, C)
            elif isinstance(out, dict) and 'x_norm_patchtokens' in out:
                x = out['x_norm_patchtokens']
            else:
                # Fallback: assume (B, N+1, C) tensor named 'out'
                x = out[:, 1:, :]
            B, N, C = x.shape
            Ht = Wt = int(math.sqrt(N))
            tokens.append(x.transpose(1,2).reshape(B, C, Ht, Wt))

        h = model.forward_features.register_forward_hook(hook_feats)
        _ = model(x)
        h.remove()
        feats = tokens[0]
        return F.normalize(feats, dim=1)

    else:
        raise ValueError("Unknown backbone id")

# ----------------------------
# 3) Matching utilities
# ----------------------------
def cosine_sim_map(featsA, featsB):
    """
    featsA, featsB: (1,C,Ht,Wt)
    Returns sim: (HtA*WtA, HtB*WtB) cosine similarity matrix
    """
    B, C, Ha, Wa = featsA.shape
    _, _, Hb, Wb = featsB.shape
    A = featsA.view(C, Ha*Wa).T  # (Na, C)
    Bf = featsB.view(C, Hb*Wb).T # (Nb, C)
    # (Na, C) x (C, Nb) -> (Na, Nb)self.img_size
    sim = torch.matmul(A, Bf.T)
    return sim  # already normalized, so dot == cosine

def grid_argmax_matches(sim, Ha, Wa, Hb, Wb):
    """
    sim: (Na, Nb), Na=Ha*Wa, Nb=Hb*Wb
    Returns (Ha,Wa,2) with (xb,yb) in token coordinates
    """
    Na, Nb = sim.shape
    idx = sim.argmax(dim=1)                 # (Na,)
    yb = (idx // Wb).view(Ha, Wa)           # row in B
    xb = (idx %  Wb).view(Ha, Wa)           # col in B
    return torch.stack([xb, yb], dim=-1)    # (Ha,Wa,2)

def points_to_token_coords(points_xy, img_size, patch_stride):
    """
    points_xy: (N,2) pixel coords in original resized image (x,y)
    Returns integer token coords (xt, yt)
    """
    pts = np.asarray(points_xy)
    xt = np.clip(pts[:,0] / patch_stride, 0, img_size[0]//patch_stride - 1)#.astype(int)
    yt = np.clip(pts[:,1] / patch_stride, 0, img_size[1]//patch_stride - 1)#.astype(int)
    return np.stack([xt, yt], axis=1)

def token_to_pixel_coords(token_xy, patch_stride):
    # center of the patch
    xy = np.asarray(token_xy)
    px = (xy[:,0] + 0.0) * patch_stride
    py = (xy[:,1] + 0.0) * patch_stride
    return np.stack([px, py], axis=1)

# ----------------------------
# 4) Public API
# ----------------------------
class RawDinoMatcher:
    def __init__(self, org_img_size = [640, 480], device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.org_img_size = org_img_size
        self.model, self.inp_size, self.patch_stride, self.feat_dim, self.preprocess, self.backbone = load_dino(self.device)

    def _prep(self, img):
        pil = Image.fromarray(img).convert("RGB")
        x = self.preprocess(pil).unsqueeze(0).to(self.device)  # (1,3,H,W)
        feats = extract_patch_tokens(self.model, x, self.backbone)  # (1,C,Ht,Wt) L2-normed
        feats = torch.nn.functional.interpolate(feats, size = img.shape[:2], mode = 'bilinear')
        return feats

    @torch.no_grad()
    def dense_match(self, imgA, imgB):
        """
        Returns:
          - matches_token: (Ht,Wa,2) predicted coords in B per token in A
          - sim (optional): (Na, Nb) similarity matrix (on CPU, float32)
          - meta: dict with sizes
        """
        fa = self._prep(imgA)
        fb = self._prep(imgB)
        _, _, Ha, Wa = fa.shape
        _, _, Hb, Wb = fb.shape
        sim = cosine_sim_map(fa, fb)  # (Na,Nb)
        matches_token = grid_argmax_matches(sim, Ha, Wa, Hb, Wb)
        return matches_token.cpu().numpy(), sim.cpu().float().numpy(), {
            "Ha": Ha, "Wa": Wa, "Hb": Hb, "Wb": Wb,
            "patch_stride": self.patch_stride, "img_size": self.img_size
        }

    @torch.no_grad()
    def track_points(self, imgA, imgB, _points_xy):
        """
        points_xy: list/array of (x,y) pixel coords **after resize** to img_size.
        Returns matched pixel coords in imgB.
        """
        points_xy = _points_xy * (np.array(self.inp_size)/self.org_img_size)
        fa = self._prep(imgA)
        fb = self._prep(imgB)
        _, _, Ha, Wa = fa.shape
        _, _, Hb, Wb = fb.shape
        sim = cosine_sim_map(fa, fb)  # (Na,Nb)

        pts_tok = points_to_token_coords(points_xy, self.inp_size, self.patch_stride)  # (N,2)
        idx_a  = (pts_tok[:,1].astype(int) * Wa + pts_tok[:,0].astype(int))  # flatten index in A
        idx_a1  = np.floor(idx_a).astype(int)
        idx_a2  = np.ceil(idx_a).astype(int)
        idx_af  = idx_a - idx_a1
        
        best_b1 = sim[idx_a1].argmax(dim=1).cpu().numpy()  # (N,)
        best_b2 = sim[idx_a2].argmax(dim=1).cpu().numpy()  # (N,)
        best_b  = best_b2 * idx_af + best_b1 * (1- idx_af)
        
        best_c = sim[idx_a].argmax(dim=1).cpu().numpy()  # (N,)
        #import pdb; pdb.set_trace()
        xt = best_b % Wb
        yt = best_b / Wb
        matched_tok = np.stack([xt, yt], axis=1)
        matched_px  = token_to_pixel_coords(matched_tok, self.patch_stride)
        matched_px  = matched_px * (np.array(self.org_img_size)/self.inp_size)
        
        return matched_px

# ----------------------------
# 5) Example usage
# ----------------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("-f")
    ap.add_argument("--imgA", default='/media/irfan/TRANSCEND/camera_data/stech/shanghaitech/training/frames/01_068/000.jpg')
    ap.add_argument("--imgB", default='/media/irfan/TRANSCEND/camera_data/stech/shanghaitech/training/frames/01_068/005.jpg')
    ap.add_argument("--points", type=str, default=None,
                    help="Comma-separated pixel points in A, e.g. '100:150,200:120' (x:y after resize)'")
    args = ap.parse_args()

    matcher = RawDinoMatcher()
    if args.points is None:
        matches_token, _, meta = matcher.dense_match(args.imgA, args.imgB)
        # Convert token matches to pixel coordinates for visualization if you like:
        px = token_to_pixel_coords(matches_token.reshape(-1,2), meta["patch_stride"])
        print(f"[dense] token grid: A={meta['Wa']}x{meta['Ha']} -> B={meta['Wb']}x{meta['Hb']}")
        print("Example first 10 matched pixels in B:\n", px[:10])
    else:
        pts = []
        for p in args.points.split(","):
            x,y = p.split(":")
            pts.append([float(x), float(y)])
        pts = np.asarray(pts, dtype=float)
        # Ensure your provided points correspond to the model's resized size (img_size x img_size)
        out = matcher.track_points(args.imgA, args.imgB, pts)
        print("[points] matched pixel coords in B (approx):")
        for (x,y) in out:
            print(f"{x:.1f},{y:.1f}")
