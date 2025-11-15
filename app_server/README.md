```md
README - Servicio de inferencia simple (lista para Render)
==========================================================

Qué hace
--------
Recibe una imagen (POST /analyze) y devuelve estimaciones de:
- elasticidad
- arrugas generales
- arrugas profundas
- líneas de expresión
- pigmentación
- edad biológica (opcional, si pones model_weights.pth)

Cómo usar (resumen para móvil)
1. Crea un repositorio en GitHub y añade estos archivos (main.py, model_utils.py, requirements.txt, Dockerfile, .gitignore, README.md).
2. En Render.com crea un "New Web Service", conectas GitHub, eliges el repo y despliegas. Render detectará el Dockerfile y hará el resto.
3. Tu servicio quedará en una URL pública tipo: https://tu-servicio.onrender.com
4. En tu página WordPress pega el snippet que te doy y sustituye SERVER_URL por https://tu-servicio.onrender.com/analyze

Notas importantes
- Seguridad: en producción restringe CORS a tu dominio.
- Privacidad: pide consentimiento y evita guardar imágenes si no quieres.
- Si quieres que estime edad, coloca un modelo PyTorch llamado model_weights.pth junto a main.py.
