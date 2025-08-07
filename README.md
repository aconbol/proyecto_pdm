# 🏭 Sistema de Mantenimiento Predictivo con Machine Learning
## Predicción de Fallas en Moto-Compresores - Sector Oil & Gas

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-green.svg)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Status-In%20Development-yellow.svg)]()

---

## 📋 Descripción del Proyecto

Este proyecto de **tesis de grado** desarrolla un sistema integral de **mantenimiento predictivo** utilizando técnicas avanzadas de **Machine Learning** para moto-compresores industriales en el sector Oil & Gas. El sistema implementa clasificación binaria para predecir fallas con 7 días de anticipación, combinando análisis de series temporales, ingeniería de características avanzada y modelos de ML optimizados.

### 🎓 **Contexto Académico**
- **Estudiante**: Miguel Salazar  
- **Institución**: Universidad EMI  
- **Programa**: Trabajo de Grado en Ingeniería  
- **Enfoque**: Machine Learning aplicado a mantenimiento predictivo industrial

### 🏭 **Aplicación Industrial**
- **Sector**: Oil & Gas (Hidrocarburos)
- **Equipos**: Moto-compresores críticos
- **Objetivo**: Reducir fallas no programadas y optimizar mantenimiento

---

## 🎯 Objetivos del Sistema

### **Objetivos Técnicos**
- 🔮 **Predicción temprana de fallas** con ventana de 7 días de anticipación
- 📊 **Clasificación binaria optimizada** (Normal vs Pre-falla)
- ⚖️ **Balance precision-recall** apropiado para equipos críticos
- 🚀 **Sistema deployable** en entornos industriales

### **Objetivos Operacionales** 
- 💰 **Reducir costos operativos** en 30-50% (target: $15-30M anuales)
- ⬆️ **Aumentar disponibilidad** de equipos en 20-40% 
- 🛡️ **Mejorar seguridad operacional** eliminando fallas catastróficas
- ⚡ **Optimizar programación** de mantenimientos preventivos

### **Objetivos Académicos**
- 📚 Desarrollar **marco metodológico** para ML en mantenimiento industrial
- 🔬 Contribuir al estado del arte en **series temporales industriales**
- 📖 Generar **conocimiento reproducible** para la comunidad científica

---

## 🏗️ Estructura del Proyecto

### **📁 Organización de Archivos**

```
proyecto_pdm/
├── 📊 01_exploratory_data_analysis.ipynb     # Análisis exploratorio completo
├── 🔧 02_data_preprocessing.ipynb            # Preprocesamiento de datos
├── ⚙️ 03_feature_engineering.ipynb           # Ingeniería de características
├── 🤖 04_model_training.ipynb                # Entrenamiento y evaluación integral
├── 📁 data/                                  # Repositorio de datos
│   ├── raw/                                  # 28 archivos Excel originales
│   │   ├── *2023*.xls                        # 12 archivos históricos (2023)
│   │   └── *2024-2025*.xlsx                  # 16 archivos operacionales
│   ├── processed/                            # Datasets procesados y optimizados
│   │   ├── timeseries_data_temporal_fixed.parquet
│   │   ├── featured_dataset_*.parquet
│   │   └── featured_dataset_with_target.parquet
│   └── models/                               # Modelos entrenados serializados
│       ├── modelo_mantenimiento_predictivo.joblib
│       └── modelo_metadatos.json
├── 📄 documentos/                           # Documentación del proyecto
│   ├── Plan_Mejoras_Modelos_ML_Academico.md # Plan técnico detallado
│   ├── investigación PdM Moto-Compresor v1.0.docx
│   └── mp 1v2 salazar_rev3.0.docx
├── 📋 eventos/                              # Historial de eventos críticos
│   └── Historial C1 RGD.xlsx               # Eventos de falla documentados
├── 🔧 crear_variable_objetivo.py            # Script para generación de target
├── 📄 CLAUDE.md                             # Documentación técnica para IA
└── 📖 README.md                             # Este archivo
```

### **📊 Datasets y Volumen de Datos**

- **📈 Total de observaciones**: 19,752 registros temporales
- **🔢 Características finales**: 144 variables derivadas  
- **⏰ Período de datos**: Enero 2023 - Abril 2025
- **📁 Tamaño total**: ~60MB de datos procesados
- **🎯 Variable objetivo**: Clasificación binaria (Normal/Pre-falla)
- **⚖️ Balance de clases**: 59.9% Normal, 40.1% Pre-falla

---

## 🔬 Pipeline de Machine Learning Implementado

### **Metodología CRISP-DM Adaptada**

El proyecto implementa una metodología rigurosa de ciencia de datos adaptada para sistemas industriales críticos:

### **1. 📊 Análisis Exploratorio de Datos (EDA)**
   **✅ COMPLETADO** - Notebook `01_exploratory_data_analysis.ipynb`
   - 🔍 **Metodología estructurada**: 7 fases de análisis secuencial
   - 📁 **Carga inteligente**: Funciones `load_sensor_file()` y `smart_load_and_combine_sensors()`
   - 🎯 **Variables críticas**: Identificación automática (COMPRESOR, MOTOR, temperatura, presión)
   - 📈 **Análisis temporal**: Medias móviles (24H, 168H, 720H)
   - 🚨 **Detección de anomalías**: Método IQR con límites dinámicos
   - 📊 **Visualizaciones**: Plotly interactivo + Seaborn estático

### **2. 🔧 Preprocesamiento de Datos**
   **✅ COMPLETADO** - Notebook `02_data_preprocessing.ipynb`
   - 🧹 **Limpieza avanzada**: Validación automática de calidad de datos
   - 🔄 **Consolidación temporal**: Unificación de 28 archivos Excel
   - ⏰ **Índice temporal**: Creación de timestamps consistentes
   - 📏 **Normalización**: Preparación para ingeniería de características
   - 💾 **Optimización**: Formato Parquet para eficiencia de almacenamiento

### **3. ⚙️ Ingeniería de Características**
   **✅ COMPLETADO** - Notebook `03_feature_engineering.ipynb`
   - 🔄 **Rolling Features**: Estadísticas móviles (media, std, min, max) en ventanas 6H, 24H, 72H
   - ⏪ **Lag Features**: Variables retardadas (2H, 12H, 48H) para memoria temporal
   - 📊 **Diferencias temporales**: Cambios y tasas de variación
   - 🎯 **Variable objetivo**: Creación de etiquetas binarias basada en eventos reales
   - 📈 **Dataset final**: 144 características derivadas de ingeniería avanzada

### **4. 🤖 Entrenamiento y Evaluación Integral** 
   **✅ COMPLETADO** - Notebook `04_model_training.ipynb`
   - ⏰ **División cronológica**: Time-based split (80/20) para evitar data leakage
   - 🔧 **Pipeline robusto**: SimpleImputer → StandardScaler → Clasificador
   - 🤖 **Modelos implementados**: Random Forest + Logistic Regression
   - ⚖️ **Manejo de desbalance**: `class_weight='balanced'` automático
   - 📊 **Evaluación rigurosa**: Múltiples métricas + análisis costo-beneficio
   - 🔍 **Interpretabilidad**: Feature importance + análisis de matrices de confusión
   - 💾 **Serialización**: Modelos listos para despliegue con metadatos completos
   - 📈 **Conclusiones científicas**: Análisis académico detallado de resultados

---

## 🛠️ Stack Tecnológico Implementado

### **🐍 Core de Machine Learning**
- **Python 3.8+**: Lenguaje principal de desarrollo
- **Jupyter Notebooks**: Entorno de desarrollo interactivo y documentación
- **Scikit-learn 1.0+**: Algoritmos de ML y pipelines
- **NumPy**: Computación numérica optimizada
- **Pandas**: Manipulación y análisis de datos estructurados

### **📊 Análisis de Datos y Visualización**
- **Matplotlib & Seaborn**: Visualizaciones estadísticas estáticas
- **Plotly**: Gráficos interactivos para análisis exploratorio
- **SciPy**: Funciones científicas avanzadas (FFT, estadística)

### **⚙️ Procesamiento de Datos**
- **xlrd & openpyxl**: Lectura de archivos Excel (.xls/.xlsx)
- **PyArrow**: Formato Parquet para optimización de almacenamiento
- **SimpleImputer**: Manejo robusto de valores NaN en features temporales

### **🤖 Modelos de Machine Learning**
- **Random Forest Classifier**: Ensemble de árboles con feature importance
- **Logistic Regression**: Modelo lineal con interpretabilidad
- **StandardScaler**: Normalización de características
- **Pipeline**: Flujo integrado de preprocesamiento y modelado

---

## 📊 Resultados Obtenidos del Sistema

### **🤖 Modelos Implementados y Evaluados**

#### **🏆 Random Forest Classifier** - *Modelo Seleccionado*
- **Configuración**: 100 estimadores, max_depth=10, class_weight='balanced'
- **Estrategia**: Conservadora (prioriza detección de fallas sobre precisión)
- **Justificación**: Excelente para aplicaciones de seguridad crítica

#### **📊 Logistic Regression** - *Baseline Comparativo* 
- **Configuración**: class_weight='balanced', solver='liblinear'
- **Estrategia**: Modelo lineal interpretable
- **Uso**: Línea base para comparación de rendimiento

### **📈 Métricas de Rendimiento Obtenidas**

| Modelo | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|--------|----------|-----------|--------|----------|---------|
| **Random Forest** | 49.4% | 7.8% | **100.0%** ✅ | 14.5% | **0.877** ✅ |
| **Logistic Regression** | 56.3% | 5.1% | 52.1% | 9.3% | 0.584 |

### **🎯 Análisis Crítico de Resultados**

#### **✅ Fortalezas Identificadas**
- **🚨 Recall Perfecto (RF)**: Detecta el 100% de fallas reales (0 falsos negativos)
- **📊 AUC-ROC Excelente (RF)**: 0.877 indica buena capacidad de discriminación
- **⚖️ Estrategia conservadora**: Apropiada para equipos críticos del sector O&G
- **🔧 Pipeline robusto**: Manejo automático de valores NaN con SimpleImputer

#### **❌ Limitaciones Críticas**
- **🚨 Precisión muy baja**: 7.8% (RF) implica 92% de falsas alarmas
- **📉 F1-Score insuficiente**: 0.145 (RF) inadecuado para producción industrial
- **⚖️ Desbalance temporal**: División cronológica resultó en ratio extremo 22.4:1
- **🎯 Ventana de predicción**: 7 días puede ser insuficiente para patrones complejos

### **💰 Análisis Costo-Beneficio**

**Costos operacionales estimados por período de evaluación:**
- **Random Forest**: $1,999,000 (solo falsas alarmas, 0 fallas perdidas)
- **Logistic Regression**: $5,695,000 (falsas alarmas + fallas críticas perdidas)
- **💡 Ahorro con RF**: $3,696,000 por evitar fallas catastróficas

### **🚨 Veredicto Científico Actual**

**❌ Los modelos NO son confiables para despliegue en producción**

**Justificación técnica:**
- F1-Score < 0.15 indica balance precision-recall inadecuado
- Precisión < 0.10 resulta en falsas alarmas operacionalmente inviables
- Requiere optimización significativa antes de implementación industrial

---

## 🚀 Plan de Mejoras y Próximos Pasos

### **📋 Roadmap de Optimización**

Basándose en el análisis científico realizado, se ha desarrollado un **Plan de Mejoras estructurado** documentado en:
📄 `documentos/Plan_Mejoras_Modelos_ML_Academico.md`

#### **🎯 Track A: Optimización de Modelos Tradicionales** *(3-4 semanas)*
**Objetivo**: Sistema viable para piloto industrial
- **Fase 1** (3-5 días): Ajuste de hiperparámetros + optimización de umbral
- **Fase 2** (1-2 semanas): SMOTE + validación cruzada temporal
- **Fase 3** (2-3 semanas): Ingeniería de características avanzada + ensemble
- **Meta**: F1-Score > 0.55, Precision > 0.25

#### **🤖 Track B: Deep Learning** *(2-4 meses)*
**Objetivo**: Sistema de próxima generación
- **Mes 1-2**: Implementación de arquitecturas LSTM
- **Mes 2-3**: Transformers para series temporales
- **Mes 3-4**: Arquitecturas híbridas CNN-LSTM
- **Meta**: F1-Score > 0.70, Precision > 0.35

### **🎖️ Valor Científico y Contribuciones**

#### **🔬 Contribuciones Académicas**
- ✅ **Framework metodológico** para ML en mantenimiento predictivo industrial
- ✅ **Análisis riguroso** de trade-offs precision vs recall en sistemas críticos  
- ✅ **Solución técnica** para manejo de valores NaN en features temporales
- ✅ **Pipeline reproducible** de ciencia de datos para series temporales industriales

#### **💼 Impacto Empresarial Proyectado**
- 💰 **Ahorro potencial**: $15-30 millones USD anuales por equipo
- 📈 **Mejora en disponibilidad**: 20-40% reducción en downtime no programado  
- 🛡️ **Seguridad operacional**: Eliminación de fallas catastróficas
- ⚡ **Optimización de mantenimiento**: 25-40% eficiencia mejorada

---

## 🛠️ Guía de Instalación y Uso

### **📋 Prerrequisitos**
- **Python 3.8+** con pip
- **RAM**: 16GB mínimo (32GB recomendado para mejoras futuras)
- **Almacenamiento**: 2GB libres para datos y modelos

### **⚡ Instalación Rápida**

```bash
# 1. Clonar repositorio
git clone <url-del-repositorio>
cd proyecto_pdm

# 2. Crear entorno virtual
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# 3. Instalar dependencias
pip install pandas numpy scikit-learn matplotlib seaborn plotly xlrd openpyxl jupyter

# 4. Ejecutar Jupyter
jupyter notebook
```

### **📖 Ejecución del Pipeline**

#### **🔄 Flujo Secuencial OBLIGATORIO**

```bash
# Ejecutar notebooks EN ORDEN (dependencias entre ellos)
jupyter notebook 01_exploratory_data_analysis.ipynb      # EDA completo
jupyter notebook 02_data_preprocessing.ipynb             # Limpieza de datos  
jupyter notebook 03_feature_engineering.ipynb            # Ingeniería de características
jupyter notebook 04_model_training.ipynb                 # Entrenamiento + evaluación
```

#### **📊 Datos Requeridos**
El sistema requiere los siguientes archivos:
- **28 archivos Excel** de datos operacionales en `data/raw/` (formato .xls/.xlsx)
- **1 archivo Excel** de historial de eventos en `eventos/Historial C1 RGD.xlsx`

#### **⚙️ Configuración Personalizable**
Modificar variables en **CLAUDE.md** para ajustar:
- Ventanas temporales para rolling features
- Períodos de lag para features temporales  
- Parámetros de detección de anomalías
- Configuración de modelos de ML

---

## 📈 Estado del Proyecto y Documentación

### **📊 Estado Actual** 
🟡 **EN DESARROLLO ACTIVO** - Fase de Optimización

- ✅ **Pipeline base completado** (4 notebooks funcionales)
- ✅ **Modelos base entrenados** (Random Forest + Logistic Regression)
- ✅ **Análisis científico riguroso** realizado
- 🔄 **Fase de mejoras** en progreso (según plan técnico)
- 📋 **Documentación académica** completa

### **📚 Documentación Técnica Disponible**

#### **📄 Documentos Principales**
- **`README.md`**: Documentación general del proyecto (este archivo)
- **`documentos/Plan_Mejoras_Modelos_ML_Academico.md`**: Plan técnico riguroso para revisores

#### **💾 Artefactos Generados**
- **Pipeline de datos**: Procesamiento automático de 28 archivos Excel
- **Modelos serializados**: Random Forest optimizado listo para despliegue
- **Dataset final**: 19,752 muestras con 144 características derivadas
- **Análisis de interpretabilidad**: Feature importance y análisis costo-beneficio

### **🔍 Características Técnicas del Sistema**

#### **📊 Variables de Entrada** *(144 características)*
- **Variables originales**: RPM, presión, temperatura, vibración (sensores críticos)
- **Rolling features**: Estadísticas móviles (6H, 24H, 72H) para captura de tendencias
- **Lag features**: Variables retardadas (2H, 12H, 48H) para memoria temporal
- **Diferencias temporales**: Cambios y tasas de variación para detección de anomalías

#### **🎯 Variable Objetivo**
- **Tipo**: Clasificación binaria (0: Normal, 1: Pre-falla)
- **Ventana de predicción**: 7 días de anticipación a falla
- **Fuente**: 108 eventos reales de falla documentados
- **Balance**: 59.9% operación normal, 40.1% pre-falla

#### **⚙️ Pipeline de ML**
- **Preprocessing**: SimpleImputer (mediana) → StandardScaler
- **División temporal**: 80/20 cronológica (sin data leakage)
- **Validación**: Métricas robustas para clasificación desbalanceada
- **Serialización**: Modelos listos para producción con metadatos

---

## 🎓 Información Académica y Contacto

### **👨‍🎓 Autor**
- **Estudiante**: Miguel Salazar
- **Institución**: Escuela Militar de Ingeniería, EMI
- **Programa**: Trabajo de Grado en Ingeniería
- **Área de investigación**: Machine Learning aplicado a mantenimiento predictivo

### **🏆 Supervisión Académica**
- **Tutor de Trabajo de Grado**: Ing. Angel Contreras Joffre
- **Comité académico**: Tribunal de Grado
- **Carrera**: Ingeniería Mecatrónica
- **Institución**: Escuela Militar de Ingeniería, EMI

### **📚 Citas y Referencias**

Si utiliza este trabajo en investigación académica, favor citar como:
```
Salazar, M. (2025). Sistema de Mantenimiento Predictivo con Machine Learning 
para Moto-Compresores Industriales. Trabajo de Grado, Universidad EMI.
```

### **🤝 Colaboración Académica**

Este proyecto está abierto a:
- ✅ **Colaboración académica** y científica
- ✅ **Validación por expertos** del sector industrial
- ✅ **Extensión a otros equipos** rotativos críticos
- ✅ **Mejoras metodológicas** y técnicas

---

## ⚠️ Consideraciones Importantes

### **🚨 Advertencias de Uso**

- **Decisiones críticas**: Siempre validar predicciones con expertos antes de mantenimiento
- **Entorno controlado**: Sistema actual no certificado para producción industrial
- **Validación continua**: Requiere monitoreo de performance en datos nuevos
- **Expertise humana**: Complementa pero no reemplaza experiencia de ingenieros

### **🔒 Responsabilidad y Disclaimers**

- Este sistema es **experimental y académico**
- Los resultados actuales requieren **optimización adicional**
- No se garantiza rendimiento en **condiciones operacionales diferentes**
- El uso en **equipos críticos** requiere validación exhaustiva adicional

### **📊 Estado Desarrollo**

🟡 **PROYECTO EN DESARROLLO ACTIVO**
- **Fase actual**: Optimización de modelos base
- **Próxima etapa**: Implementación de mejoras propuestas
- **Meta a corto plazo**: Sistema viable para piloto industrial
- **Meta a largo plazo**: Despliegue en entorno de producción

---

*Última actualización: Agosto 2025 - Versión 1.0 del sistema base completada*