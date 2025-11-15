"""
model_utils.py - utilidades simples para detectar cara y calcular proxies de piel.
Esta versión usa facenet-pytorch (MTCNN) para detectar la cara.
Si no quieres usar modelo de edad, NO necesitas model_weights.pth.
"""

from facenet_pytorch import MTCNN
from PIL import Image
import numpy as np
from skimage import color, filters
import torch, os, math

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
_mtcnn = MTCNN(keep_all=True, device=DEVICE, post_process=False)

def load_age_model(model_path: str = None):
    if model_path is None or not os.path.exists(model_path):
        print("No se encontró modelo de edad (modelo opcional).")
        return None
    try:
        model = torch.load(model_path, map_location=DEVICE)
        model.eval()
        print("Modelo de edad cargado desde", model_path)
        return model
    except Exception as e:
        print("Error cargando modelo:", e)
        return None

def pil_to_np(img: Image.Image):
    return np.asarray(img)

def crop_safe(img_np, x, y, w, h):
    H, W = img_np.shape[:2]
    x0 = max(0, int(x))
    y0 = max(0, int(y))
    x1 = min(W, int(x + w))
    y1 = min(H, int(y + h))
    if x1 <= x0 or y1 <= y0:
        return None
    return img_np[y0:y1, x0:x1]

def grayscale(img_np):
    if img_np.ndim == 3:
        return color.rgb2gray(img_np) * 255.0
    return img_np.astype(float)

def wrinkle_score(patch_np):
    gray = grayscale(patch_np)
    lap = filters.laplace(gray, ksize=3)
    score = float(np.mean(np.abs(lap)))
    val = min(100.0, (score / 12.0) * 100.0)
    return float(val)

def deep_wrinkle_score(patch_np):
    gray = grayscale(patch_np)
    sob = np.hypot(filters.sobel_h(gray), filters.sobel_v(gray))
    count = float(np.sum(sob > 0.12 * 255.0))
    density = count / sob.size
    val = min(100.0, density * 1500.0)
    return float(val)

def pigmentation_score(patch_np):
    rgb = patch_np.astype(np.uint8)
    lab = color.rgb2lab(rgb)
    L = lab[:, :, 0]
    a = lab[:, :, 1]
    mask = (L < 55) & (a > 5)
    density = float(np.sum(mask)) / mask.size
    val = min(100.0, density * 1500.0)
    return float(val)

def expression_lines_score(patch_np):
    gray = grayscale(patch_np)
    sob = np.hypot(filters.sobel_h(gray), filters.sobel_v(gray))
    val = float(np.mean(sob)) / 255.0 * 100.0
    return float(min(100.0, val * 1.5))

def elasticity_score(landmarks, face_box, left_cheek_patch_np):
    try:
        rug = wrinkle_score(left_cheek_patch_np)
        nose = landmarks.get('nose', None)
        jaw = landmarks.get('jaw', None)
        if nose is None or jaw is None:
            sag = 0.4
        else:
            nx, ny = nose
            jx, jy = jaw
            face_w = face_box[2]
            sag = math.hypot(nx - jx, ny - jx) / max(1.0, face_w)
        elasticity = 100 - (0.65 * rug) - (sag * 45.0)
        elasticity = max(0.0, min(100.0, elasticity))
        return float(elasticity)
    except Exception:
        return 50.0

def detect_face_and_landmarks(img_pil):
    boxes, probs, landmarks = _mtcnn.detect(img_pil, landmarks=True)
    if boxes is None or len(boxes) == 0:
        return None
    areas = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes]
    idx = int(np.argmax(areas))
    b = boxes[idx]
    lm = landmarks[idx]
    x0, y0, x1, y1 = b
    box = (float(x0), float(y0), float(x1 - x0), float(y1 - y0))
    mouth_center = ((lm[3][0] + lm[4][0]) / 2.0, (lm[3][1] + lm[4][1]) / 2.0)
    jaw_point = (mouth_center[0], y1)
    landmarks_dict = {
        'left_eye': tuple(lm[0].tolist()),
        'right_eye': tuple(lm[1].tolist()),
        'nose': tuple(lm[2].tolist()),
        'mouth_left': tuple(lm[3].tolist()),
        'mouth_right': tuple(lm[4].tolist()),
        'jaw': jaw_point
    }
    return {'box': box, 'landmarks': landmarks_dict}

def preprocess_for_age_model(face_patch_pil, target_size=224):
    img = face_patch_pil.resize((target_size, target_size))
    arr = np.array(img).astype(np.float32) / 255.0
    arr = np.transpose(arr, (2,0,1))
    import torch
    tensor = torch.from_numpy(arr).unsqueeze(0).to(DEVICE)
    return tensor

def predict_age(face_patch_pil, model):
    if model is None:
        return None
    try:
        t = preprocess_for_age_model(face_patch_pil, target_size=224)
        with torch.no_grad():
            out = model(t)
        import numpy as _np
        if hasattr(out, 'cpu'):
            val = out.cpu().numpy().squeeze().item()
        else:
            val = float(out)
        return float(val)
    except Exception as e:
        print("Error predicción edad:", e)
        return None

def analyze_image_pil(img_pil, age_model=None):
    img_np = np.array(img_pil)
    detected = detect_face_and_landmarks(img_pil)
    if detected is None:
        return {'error': 'No se detectó cara. Asegúrate de que la foto muestre la cara visible.'}
    box = detected['box']
    lm = detected['landmarks']
    left_eye = lm['left_eye']
    right_eye = lm['right_eye']
    left_cheek_center = (left_eye[0] - 30, left_eye[1] + 40)
    right_cheek_center = (right_eye[0] + 30, right_eye[1] + 40)

    def center_to_rect(center, size=120):
        cx, cy = center
        return (int(cx - size/2), int(cy - size/2), size, size)

    left_rect = center_to_rect(left_cheek_center, size=120)
    right_rect = center_to_rect(right_cheek_center, size=120)

    face_x, face_y, face_w, face_h = box
    chest_rect = (int(face_x + face_w*0.12), int(face_y + face_h + 8), int(face_w*0.76),
                  int(min(240, img_np.shape[0] - (face_y + face_h + 8))))

    patches = {}
    patches['left_cheek'] = crop_safe(img_np, *left_rect)
    patches['right_cheek'] = crop_safe(img_np, *right_rect)
    patches['chest'] = crop_safe(img_np, *chest_rect)
    if patches['left_cheek'] is None:
        patches['left_cheek'] = crop_safe(img_np, face_x + face_w*0.15, face_y + face_h*0.3, 120, 120)
    if patches['right_cheek'] is None:
        patches['right_cheek'] = crop_safe(img_np, face_x + face_w*0.55, face_y + face_h*0.3, 120, 120)

    wrinkle_left = wrinkle_score(patches['left_cheek'])
    wrinkle_right = wrinkle_score(patches['right_cheek'])
    wrinkles_general = float((wrinkle_left + wrinkle_right) / 2.0)

    deep_left = deep_wrinkle_score(patches['left_cheek'])
    deep_right = deep_wrinkle_score(patches['right_cheek'])
    wrinkles_deep = float((deep_left + deep_right) / 2.0)

    pigmentation_left = pigmentation_score(patches['left_cheek'])
    pigmentation_right = pigmentation_score(patches['right_cheek'])
    pigmentation = float((pigmentation_left + pigmentation_right) / 2.0)

    expr = expression_lines_score(patches['left_cheek'])
    elasticity = elasticity_score(lm, box, patches['left_cheek'])

    fx, fy, fw, fh = int(face_x), int(face_y), int(face_w), int(face_h)
    face_patch_np = crop_safe(img_np, fx, fy, fw, fh)
    age_pred = None
    if age_model is not None and face_patch_np is not None:
        from PIL import Image as PILImage
        face_patch_pil = PILImage.fromarray(face_patch_np)
        age_pred = predict_age(face_patch_pil, age_model)

    return {
        'elasticity': round(float(elasticity), 2),
        'wrinkles_general': round(float(wrinkles_general), 2),
        'wrinkles_deep': round(float(wrinkles_deep), 2),
        'expression_lines': round(float(expr), 2),
        'pigmentation': round(float(pigmentation), 2),
        'age_biological': None if age_pred is None else round(float(age_pred), 2),
        'debug': {
            'face_box': [fx, fy, fw, fh],
            'left_cheek_rect': list(left_rect),
            'right_cheek_rect': list(right_rect),
            'chest_rect': list(chest_rect)
        }
    }
