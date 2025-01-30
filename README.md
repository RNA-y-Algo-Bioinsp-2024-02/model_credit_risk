# Model Credit Risk

Este repositorio contiene el código para realizar tareas de preprocesamiento, ingeniería de características y la creación de un modelo de riesgo crediticio. A continuación, se detallan los componentes principales y su funcionalidad:

---

## Contenido del repositorio

1. **Preprocesamiento e ingeniería de características**  
   - Scripts y notebooks utilizados para la limpieza y transformación del dataset.
   - Creación de nuevas características que mejoran el poder predictivo del modelo.
   - Preparación final del conjunto de datos para el entrenamiento.

2. **Creación del modelo**  
   - Entrenamiento del modelo final denominado `modelo_final_bueno.h5`.
   - Ajuste de hiperparámetros y validación de la performance para garantizar la calidad del modelo.

3. **API y despliegue**  
   - El archivo `app.py` contiene la lógica para exponer el modelo a través de una API, permitiendo su consumo desde la aplicación web.
   - Se utiliza **GitHub Actions** para automatizar el proceso de despliegue en cada commit realizado al repositorio.

---

## Estructura de archivos y directorios

```
model_credit_risk/
├── data/                     # Dataset y archivos de preprocesamiento
├── notebooks/               # Notebooks con experimentos y análisis
├── scripts/                 # Scripts de transformación de datos y entrenamiento
├── app.py                   # API para exponer el modelo entrenado
├── modelo_final_bueno.h5    # Modelo final entrenado
└── requirements.txt         # Dependencias y bibliotecas necesarias
```

---

## Cómo ejecutar la API localmente

1. **Clonar el repositorio**  
   ```bash
   git clone https://github.com/usuario/model_credit_risk.git
   ```

2. **Instalar dependencias** (recomendado en un entorno virtual)  
   ```bash
   cd model_credit_risk
   pip install -r requirements.txt
   ```

3. **Ejecutar la aplicación**  
   ```bash
   python app.py
   ```
   La aplicación se ejecutará (por defecto) en `http://127.0.0.1:5000/`.

---

## Automatización con GitHub Actions

El despliegue se encuentra configurado mediante **GitHub Actions**. Cada vez que se realiza un push o se abre un pull request, se desencadena automáticamente un flujo de trabajo que:

- Despliega la última versión de la aplicación.

---
## Blog
El acceso al reporte técnico es a través del siguiente link

[Modelos de Riesgo de Crédito](https://rna-y-algo-bioinsp-2024-02.github.io/blog/posts/modelos-riesgo-credito/)
---

## Licencia

Este proyecto se distribuye bajo la licencia [MIT](LICENSE).  

---

**Autor(es):**  
- Tomás Rodríguez Taborda
- Julian Castaño Pineda
- Catalina Restrepo Salgado
- Luis Andrés Altamar Romero
