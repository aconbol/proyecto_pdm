# 🏭 Sistema de Mantenimiento Predictivo con IA
## Moto-Compresores Industriales

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-green.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

### 📋 Descripción del Proyecto

Este proyecto implementa un sistema de **mantenimiento predictivo** utilizando técnicas de **Inteligencia Artificial** y **Machine Learning** para optimizar el mantenimiento de moto-compresores industriales. El sistema analiza datos operacionales en tiempo real para predecir fallos y programar mantenimientos preventivos, reduciendo costos operativos y aumentando la disponibilidad del equipo.

### 🎯 Objetivos

- **Predecir fallos** antes de que ocurran
- **Optimizar el mantenimiento** programado
- **Reducir costos** de reparación y tiempo de inactividad
- **Aumentar la eficiencia** operacional
- **Mejorar la seguridad** del personal

### 🏗️ Arquitectura del Proyecto

```
proyecto_pdm/
├── 📊 01_exploratory_data_analysis.ipynb    # Análisis exploratorio de datos
├── 🔧 02_data_preprocessing.ipynb           # Preprocesamiento y limpieza
├── ⚙️ 03_feature_engineering.ipynb          # Ingeniería de características
├── 🤖 04_model_training.ipynb               # Entrenamiento de modelos
├── 📈 05_model_evaluation.ipynb             # Evaluación y validación
├── 🚀 06_deployment_preparation.ipynb       # Preparación para producción
├── 📁 data/
│   ├── raw/                                 # Datos originales
│   ├── processed/                           # Datos procesados
│   └── models/                              # Modelos entrenados
└── 📄 README.md
```

### 🔬 Metodología

El proyecto sigue una metodología estructurada de **Machine Learning**:

1. **📊 Análisis Exploratorio de Datos (EDA)**
   - Exploración de la estructura y calidad de los datos
   - Identificación de patrones y tendencias
   - Detección de anomalías y valores atípicos
   - Análisis de correlaciones entre variables

2. **🔧 Preprocesamiento de Datos**
   - Limpieza y validación de datos
   - Manejo de valores faltantes
   - Normalización y estandarización
   - Creación de datasets consolidados

3. **⚙️ Ingeniería de Características**
   - Creación de características derivadas
   - Generación de indicadores de tendencia
   - Selección de características relevantes
   - Creación de variables de degradación

4. **🤖 Entrenamiento de Modelos**
   - Implementación de múltiples algoritmos (Random Forest, Gradient Boosting, SVR, etc.)
   - Optimización de hiperparámetros
   - Validación cruzada
   - Selección del mejor modelo

5. **📈 Evaluación de Modelos**
   - Análisis de rendimiento exhaustivo
   - Interpretabilidad y explicabilidad (SHAP)
   - Análisis de errores
   - Validación temporal

6. **🚀 Preparación para Despliegue**
   - Serialización del modelo
   - Creación de pipeline de inferencia
   - Validación del sistema completo
   - Documentación técnica

### 🛠️ Tecnologías Utilizadas

- **Python 3.8+**: Lenguaje principal
- **Jupyter Notebooks**: Desarrollo y documentación
- **Pandas & NumPy**: Manipulación de datos
- **Scikit-learn**: Algoritmos de Machine Learning
- **Matplotlib & Seaborn**: Visualizaciones
- **SHAP**: Explicabilidad de modelos
- **Joblib**: Serialización de modelos

### 📊 Algoritmos Implementados

- **Random Forest Regressor**
- **Gradient Boosting Regressor**
- **Support Vector Regression (SVR)**
- **Linear Regression (Ridge, Lasso)**
- **Neural Networks (MLPRegressor)**

### 🚀 Instalación y Configuración

#### Prerrequisitos

- Python 3.8 o superior
- pip (gestor de paquetes de Python)

#### Pasos de Instalación

1. **Clonar el repositorio**
   ```bash
   git clone <url-del-repositorio>
   cd proyecto_pdm
   ```

2. **Crear entorno virtual**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # En Linux/Mac
   # o
   .venv\Scripts\activate     # En Windows
   ```

3. **Instalar dependencias**
   ```bash
   pip install -r requirements.txt
   ```

4. **Ejecutar Jupyter**
   ```bash
   jupyter notebook
   ```

### 📖 Uso del Sistema

#### Ejecución Secuencial

1. **Análisis Exploratorio**: Ejecutar `01_exploratory_data_analysis.ipynb`
2. **Preprocesamiento**: Ejecutar `02_data_preprocessing.ipynb`
3. **Ingeniería de Características**: Ejecutar `03_feature_engineering.ipynb`
4. **Entrenamiento**: Ejecutar `04_model_training.ipynb`
5. **Evaluación**: Ejecutar `05_model_evaluation.ipynb`
6. **Despliegue**: Ejecutar `06_deployment_preparation.ipynb`

#### Estructura de Datos

Los datos deben estar organizados en el directorio `data/`:

```
data/
├── raw/           # Datos originales (.xls, .csv)
├── processed/     # Datos preprocesados
└── models/        # Modelos entrenados
```

### 📈 Métricas de Rendimiento

El sistema evalúa los modelos utilizando:

- **Mean Squared Error (MSE)**
- **Mean Absolute Error (MAE)**
- **R² Score**
- **Root Mean Squared Error (RMSE)**

### 🔍 Características del Modelo

#### Variables de Entrada
- Parámetros operacionales del moto-compresor
- Variables ambientales
- Historial de mantenimiento
- Indicadores de rendimiento

#### Variable Objetivo
- Tiempo hasta el próximo fallo (RUL - Remaining Useful Life)
- Probabilidad de fallo en ventana de tiempo específica

### 🎯 Resultados Esperados

- **Reducción del 30-50%** en costos de mantenimiento
- **Aumento del 20-40%** en disponibilidad del equipo
- **Detección temprana** de anomalías operacionales
- **Optimización** de la programación de mantenimientos

### 🔧 Mantenimiento del Sistema

#### Monitoreo Continuo
- Validación de la precisión del modelo
- Detección de drift en los datos
- Actualización periódica del modelo

#### Actualizaciones
- Reentrenamiento con nuevos datos
- Ajuste de hiperparámetros
- Incorporación de nuevas características

### 📝 Documentación Adicional

- **Manual Técnico**: Especificaciones técnicas detalladas
- **Guía de Usuario**: Instrucciones de uso del sistema
- **API Documentation**: Documentación de la interfaz de programación

### 🤝 Contribución

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

### 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

### 👥 Autores

- **Miguel Salazar** - *Desarrollo inicial* - [GitHub](https://github.com/tu-usuario)

### 🙏 Agradecimientos

- Equipo de mantenimiento industrial
- Expertos en moto-compresores
- Comunidad de Machine Learning

### 📞 Contacto

- **Email**: tu-email@ejemplo.com
- **LinkedIn**: [Tu Perfil](https://linkedin.com/in/tu-perfil)
- **Proyecto**: [GitHub](https://github.com/tu-usuario/proyecto_pdm)

---

**⚠️ Nota**: Este sistema está diseñado para uso industrial. Siempre valide las predicciones con expertos técnicos antes de tomar decisiones de mantenimiento críticas.

**📊 Estado del Proyecto**: En desarrollo activo 