// index.js
const express = require('express');
const multer = require('multer'); // para manejar archivos
const upload = multer();

const app = express();
const PORT = process.env.PORT || 3000;

// Ruta principal de prueba
app.get('/', (req, res) => {
  res.send('Servidor de Aging Analyzer funcionando ✅');
});

// Ruta de análisis
app.post('/analyze', upload.single('file'), (req, res) => {
  console.log("Imagen recibida:", req.file?.originalname);

  // Aquí pondrás tu lógica real de análisis de piel.
  // Por ahora devolvemos datos simulados:
  res.json({
    elasticity: 80,
    wrinkles_general: 20,
    wrinkles_deep: 15,
    expression_lines: 25,
    pigmentation: 10,
    age_biological: 35,
    debug: { filename: req.file?.originalname }
  });
});

app.listen(PORT, () => {
  console.log(`Servidor escuchando en puerto ${PORT}`);
});
