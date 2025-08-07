# Plan de Mejoras para Modelos de Machine Learning en Mantenimiento Predictivo de Moto-Compresores

**Informe Técnico para Revisión Académica**

---

**Proyecto:** Desarrollo de Sistema de Mantenimiento Predictivo para Equipos Críticos en el Sector Oil & Gas  
**Autor:** Miguel Salazar  
**Institución:** Universidad EMI  
**Fecha:** 7 de Agosto de 2025  
**Revisores:** Comité Académico de Tesis  
**Fase Actual:** Post-Entrenamiento de Modelos Base (Notebook 04)

---

## Resumen Ejecutivo

Este informe presenta un análisis crítico de los resultados obtenidos en la fase de entrenamiento de modelos de Machine Learning para mantenimiento predictivo de moto-compresores, junto con un plan estructurado de mejoras metodológicas. Los modelos base implementados (Random Forest y Logistic Regression) muestran limitaciones significativas para despliegue industrial inmediato, con F1-Scores de 0.145 y 0.092 respectivamente. Se propone un enfoque bifurcado de optimización que combina mejoras incrementales de modelos tradicionales (cronograma 3-4 semanas) con exploración de arquitecturas de Deep Learning (cronograma 2-4 meses).

**Palabras clave:** Mantenimiento Predictivo, Machine Learning, Series Temporales, Oil & Gas, Random Forest, Deep Learning

---

## 1. Introducción y Contexto

### 1.1 Problemática Industrial

La industria de Oil & Gas enfrenta desafíos críticos en el mantenimiento de equipos rotativos, donde las fallas no programadas pueden resultar en costos operacionales de $50,000-$500,000 por evento, además de riesgos de seguridad significativos. Los sistemas tradicionales de mantenimiento basados en cronogramas fijos presentan limitaciones de eficiencia y efectividad.

### 1.2 Objetivos del Sistema Propuesto

- **Objetivo Principal:** Predecir fallas en moto-compresores con 7-30 días de anticipación
- **Objetivo Secundario:** Reducir falsas alarmas a niveles operacionalmente viables (Precision > 30%)
- **Objetivo Terciario:** Mantener detección completa de fallas reales (Recall > 90%)

### 1.3 Dataset y Metodología Base

- **Datos:** 19,752 observaciones de 144 características técnicas derivadas
- **Período:** Enero 2023 - Abril 2025 (datos operacionales reales)
- **Enfoque:** Series temporales con división cronológica para evitar data leakage
- **Pipeline:** SimpleImputer → StandardScaler → Clasificador

---

## 2. Análisis de Resultados Actuales

### 2.1 Métricas de Rendimiento Obtenidas

#### 2.1.1 Random Forest Classifier
- **Accuracy:** 49.41% (sub-óptima debido a estrategia conservadora)
- **Precision:** 7.80% (92.2% de falsas alarmas)
- **Recall:** 100.00% (detección perfecta de fallas reales)
- **F1-Score:** 14.46% (balance precision-recall insuficiente)
- **AUC-ROC:** 0.877 (capacidad de discriminación excelente)

#### 2.1.2 Logistic Regression
- **Accuracy:** 56.31% (marginalmente mejor que aleatorio)
- **Precision:** 5.08% (94.9% de falsas alarmas)
- **Recall:** 52.07% (pérdida de 48% de fallas reales)
- **F1-Score:** 9.25% (rendimiento crítico)
- **AUC-ROC:** 0.584 (capacidad discriminativa marginal)

### 2.2 Matriz de Confusión - Análisis Crítico

#### Random Forest:
- **Verdaderos Negativos:** 1,783 (identificación correcta de operación normal)
- **Falsos Positivos:** 1,999 (falsas alarmas críticas para viabilidad operacional)
- **Falsos Negativos:** 0 (excelente desde perspectiva de seguridad)
- **Verdaderos Positivos:** 169 (detección completa de fallas)

### 2.3 Análisis de Costos Operacionales

Basado en estimaciones industriales:
- **Costo por falsa alarma:** $1,000 USD (mantenimiento innecesario)
- **Costo por falla perdida:** $50,000 USD (falla no programada)

**Análisis costo-beneficio:**
- Random Forest: $1,999,000 (solo falsas alarmas)
- Logistic Regression: $5,695,000 (falsas alarmas + fallas perdidas)
- **Diferencial:** Random Forest es $3,696,000 más económico

### 2.4 Diagnóstico de Problemas Fundamentales

1. **Desbalance Temporal Extremo:**
   - Entrenamiento: Ratio 1.0:1 (8,051 normales vs 7,750 pre-falla)
   - Prueba: Ratio 22.4:1 (3,782 normales vs 169 pre-falla)

2. **Limitaciones Metodológicas:**
   - División cronológica única (80/20) sin validación cruzada temporal
   - Ventana de predicción fija (7 días)
   - Ingeniería de características básica

3. **Calidad de Datos:**
   - 822 valores NaN estructurales (características temporales)
   - Concentración de eventos pre-falla en período de entrenamiento

---

## 3. Estrategia de Mejoras Propuesta

### 3.1 Enfoque Bifurcado

La propuesta metodológica contempla dos tracks paralelos de desarrollo:

#### Track A: **Optimización de Modelos Tradicionales** (Despliegue Rápido)
- **Objetivo:** Sistema viable para piloto industrial
- **Cronograma:** 3-4 semanas
- **Métricas objetivo:** F1-Score > 0.50, Precision > 0.25

#### Track B: **Exploración de Deep Learning** (Investigación Avanzada)
- **Objetivo:** Sistema de próxima generación
- **Cronograma:** 2-4 meses
- **Métricas objetivo:** F1-Score > 0.70, Precision > 0.40

---

## 4. Track A: Mejoras de Modelos Tradicionales

### 4.1 Fase 1 - Optimizaciones Inmediatas (3-5 días)

#### 4.1.1 Ajuste de Hiperparámetros
**Configuración actual vs. propuesta para Random Forest:**

| Parámetro | Actual | Propuesto | Justificación |
|-----------|--------|-----------|---------------|
| n_estimators | 100 | 300-500 | Mayor estabilidad y reducción de varianza |
| max_depth | 10 | 15-25 | Captura de patrones más complejos |
| min_samples_split | 5 | 2-3 | Mayor flexibilidad en divisiones |
| min_samples_leaf | 2 | 1 | Hojas más específicas para casos raros |
| criterion | gini | entropy | Mejor manejo de clases desbalanceadas |

**Metodología:** GridSearchCV con validación cruzada temporal (TimeSeriesSplit)

#### 4.1.2 Optimización de Umbral de Clasificación
```python
# Algoritmo propuesto
from sklearn.metrics import precision_recall_curve

def find_optimal_threshold(y_true, y_proba, cost_fp=1000, cost_fn=50000):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    
    costs = []
    for i, threshold in enumerate(thresholds):
        # Calcular matriz de confusión para cada umbral
        y_pred = (y_proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Calcular costo operacional
        total_cost = (fp * cost_fp) + (fn * cost_fn)
        costs.append(total_cost)
    
    optimal_idx = np.argmin(costs)
    return thresholds[optimal_idx]
```

**Impacto esperado:** Reducción 30-50% en costo operacional total.

### 4.2 Fase 2 - Rebalanceo y Validación (1-2 semanas)

#### 4.2.1 SMOTE (Synthetic Minority Oversampling Technique)
```python
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.pipeline import Pipeline as ImbPipeline

pipeline_smote = ImbPipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('resampler', SMOTE(random_state=42, k_neighbors=3, sampling_strategy='auto')),
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(...))
])
```

**Justificación científica:** SMOTE genera muestras sintéticas en el espacio de características, abordando el desbalance sin duplicación simple.

#### 4.2.2 Validación Cruzada Temporal
```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5, test_size=0.2, gap=24)
```

**Ventaja metodológica:** Simula múltiples escenarios de predicción temporal, mejorando la robustez estadística.

### 4.3 Fase 3 - Ingeniería de Características Avanzada (2-3 semanas)

#### 4.3.1 Características de Frecuencia
```python
def create_frequency_features(signal, fs=1.0):
    """
    Análisis espectral usando FFT para detectar patrones de frecuencia
    indicativos de degradación mecánica.
    """
    fft_values = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(signal), 1/fs)
    
    # Bandas de frecuencia críticas para maquinaria rotatoria
    power_spectrum = np.abs(fft_values)**2
    
    features = {
        'dominant_freq': freqs[np.argmax(power_spectrum[1:len(freqs)//2])],
        'spectral_centroid': np.sum(freqs[:len(freqs)//2] * power_spectrum[:len(freqs)//2]) / np.sum(power_spectrum[:len(freqs)//2]),
        'spectral_rolloff': calculate_spectral_rolloff(freqs, power_spectrum),
        'harmonic_ratio': calculate_harmonic_ratio(power_spectrum)
    }
    
    return features
```

#### 4.3.2 Características de Degradación
```python
def create_degradation_features(df, windows=[6, 12, 24, 48]):
    """
    Indicadores específicos de deterioro progresivo en equipos rotativos.
    """
    features = {}
    
    for window in windows:
        # Coeficiente de variación (estabilidad operacional)
        features[f'cv_{window}H'] = df.rolling(window).std() / df.rolling(window).mean()
        
        # Tendencia lineal (degradación gradual)
        features[f'trend_{window}H'] = df.rolling(window).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == window else np.nan
        )
        
        # Autocorrelación (cambios en patrones operacionales)
        features[f'autocorr_{window}H'] = df.rolling(window).apply(
            lambda x: x.autocorr(lag=1) if len(x) == window else np.nan
        )
    
    return features
```

### 4.4 Métricas Objetivo Track A

| Métrica | Actual | Objetivo Fase 1 | Objetivo Fase 2 | Objetivo Fase 3 |
|---------|---------|-----------------|-----------------|-----------------|
| F1-Score | 0.145 | 0.25-0.35 | 0.35-0.45 | 0.50-0.65 |
| Precision | 0.078 | 0.15-0.20 | 0.20-0.28 | 0.25-0.40 |
| Recall | 1.000 | >0.95 | >0.90 | >0.85 |
| AUC-ROC | 0.877 | >0.88 | >0.90 | >0.92 |

---

## 5. Track B: Deep Learning para Series Temporales

### 5.1 Arquitecturas Propuestas

#### 5.1.1 LSTM (Long Short-Term Memory) - Fase 1 (1-2 meses)
```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization

def create_lstm_model(sequence_length, n_features):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(sequence_length, n_features)),
        Dropout(0.2),
        BatchNormalization(),
        
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        BatchNormalization(),
        
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    return model
```

**Justificación:** LSTM maneja dependencias temporales largas, crítico para patrones de degradación gradual.

#### 5.1.2 Transformer para Series Temporales - Fase 2 (2-3 meses)
```python
import tensorflow as tf
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization

class TimeSeriesTransformer(tf.keras.Model):
    def __init__(self, d_model=128, n_heads=8, n_layers=4, n_classes=2):
        super().__init__()
        self.d_model = d_model
        
        # Multi-head attention layers
        self.attention_layers = [
            MultiHeadAttention(num_heads=n_heads, key_dim=d_model//n_heads)
            for _ in range(n_layers)
        ]
        
        # Normalization layers
        self.norm_layers = [LayerNormalization() for _ in range(n_layers)]
        
        # Classification head
        self.global_pool = tf.keras.layers.GlobalAveragePooling1D()
        self.classifier = tf.keras.layers.Dense(n_classes, activation='softmax')
    
    def call(self, inputs, training=None):
        x = inputs
        
        # Transformer blocks
        for attention, norm in zip(self.attention_layers, self.norm_layers):
            attn_output = attention(x, x, training=training)
            x = norm(x + attn_output, training=training)
        
        # Global pooling and classification
        pooled = self.global_pool(x)
        return self.classifier(pooled, training=training)
```

**Ventaja:** Mecanismo de atención permite identificar períodos críticos en secuencias temporales largas.

#### 5.1.3 Arquitectura Híbrida CNN-LSTM - Fase 3 (3-4 meses)
```python
def create_hybrid_model(sequence_length, n_features):
    # Input layer
    inputs = tf.keras.layers.Input(shape=(sequence_length, n_features))
    
    # CNN layers for local pattern extraction
    conv1 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu')(inputs)
    conv1 = tf.keras.layers.BatchNormalization()(conv1)
    conv1 = tf.keras.layers.MaxPooling1D(pool_size=2)(conv1)
    
    conv2 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu')(conv1)
    conv2 = tf.keras.layers.BatchNormalization()(conv2)
    conv2 = tf.keras.layers.MaxPooling1D(pool_size=2)(conv2)
    
    # LSTM layers for temporal dependencies
    lstm1 = tf.keras.layers.LSTM(128, return_sequences=True)(conv2)
    lstm1 = tf.keras.layers.Dropout(0.3)(lstm1)
    
    lstm2 = tf.keras.layers.LSTM(64, return_sequences=False)(lstm1)
    lstm2 = tf.keras.layers.Dropout(0.3)(lstm2)
    
    # Classification layers
    dense = tf.keras.layers.Dense(32, activation='relu')(lstm2)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(dense)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model
```

**Justificación científica:** Combinación de extracción de patrones locales (CNN) con modelado de dependencias temporales (LSTM).

### 5.2 Cronograma Deep Learning

| Fase | Duración | Modelo | Actividades Principales | Métricas Objetivo |
|------|----------|--------|------------------------|-------------------|
| 1 | 1-2 meses | LSTM | Implementación, entrenamiento, validación | F1 > 0.60 |
| 2 | 2-3 meses | Transformer | Arquitectura de atención, optimización | F1 > 0.70 |
| 3 | 3-4 meses | CNN-LSTM | Arquitectura híbrida, fine-tuning | F1 > 0.75 |

---

## 6. Metodología de Evaluación

### 6.1 Métricas de Evaluación Robustas

#### 6.1.1 Métricas Primarias
- **F1-Score Ponderado:** Balance entre precision y recall considerando costos operacionales
- **Area Under Precision-Recall Curve (AUPRC):** Más informativa que AUC-ROC para datos desbalanceados
- **Matthews Correlation Coefficient (MCC):** Métrica robusta para clasificación desbalanceada

#### 6.1.2 Métricas Secundarias
- **Specificity:** Capacidad de identificar correctamente operación normal
- **Negative Predictive Value (NPV):** Confianza en predicciones negativas
- **Balanced Accuracy:** Accuracy ajustada por desbalance de clases

### 6.2 Validación Estadística

#### 6.2.1 Validación Cruzada Temporal Anidada
```python
from sklearn.model_selection import TimeSeriesSplit

# Outer loop: Evaluación final
outer_tscv = TimeSeriesSplit(n_splits=5, test_size=0.2)

# Inner loop: Selección de hiperparámetros  
inner_tscv = TimeSeriesSplit(n_splits=3, test_size=0.2)

nested_scores = []
for train_idx, test_idx in outer_tscv.split(X):
    # Selección de modelo en inner loop
    best_model = grid_search_cv(X[train_idx], y[train_idx], cv=inner_tscv)
    
    # Evaluación en outer loop
    score = evaluate_model(best_model, X[test_idx], y[test_idx])
    nested_scores.append(score)

# Estimación imparcial del rendimiento
final_performance = np.mean(nested_scores)
confidence_interval = np.percentile(nested_scores, [2.5, 97.5])
```

#### 6.2.2 Test de Significancia Estadística
- **McNemar's Test:** Comparación entre modelos en los mismos datos de prueba
- **Wilcoxon Signed-Rank Test:** Comparación no paramétrica de rendimiento
- **Bootstrap Confidence Intervals:** Intervalos de confianza para métricas

---

## 7. Riesgos y Mitigaciones

### 7.1 Riesgos Técnicos

#### 7.1.1 Overfitting en Deep Learning
**Riesgo:** Modelos complejos pueden memorizar patrones específicos del dataset de entrenamiento.
**Mitigación:** 
- Dropout regularization (0.2-0.5)
- Early stopping con validation loss
- Cross-validation temporal estricta
- Data augmentation para series temporales

#### 7.1.2 Desbalance Persistente
**Riesgo:** Técnicas de rebalanceo pueden introducir bias artificial.
**Mitigación:**
- Evaluación con múltiples métricas robustas
- Validación en datos completamente independientes
- Análisis de sensibilidad a diferentes ratios de clases

#### 7.1.3 Drift Temporal de Datos
**Riesgo:** Cambios en patrones operacionales del equipo a lo largo del tiempo.
**Mitigación:**
- Monitoreo continuo de distribuciones de características
- Re-entrenamiento periódico del modelo
- Detección de concept drift automática

### 7.2 Riesgos Operacionales

#### 7.2.1 Latencia Computacional
**Riesgo:** Modelos Deep Learning pueden requerir tiempo de inferencia incompatible con aplicaciones en tiempo real.
**Mitigación:**
- Benchmark de tiempos de inferencia
- Optimización de modelos (pruning, quantization)
- Arquitecturas lightweight para producción

#### 7.2.2 Interpretabilidad
**Riesgo:** Modelos complejos pueden ser "cajas negras" inaceptables para aplicaciones críticas.
**Mitigación:**
- SHAP (SHapley Additive exPlanations) para explicabilidad
- LIME (Local Interpretable Model-agnostic Explanations)
- Attention weights visualization para Transformers

---

## 8. Recursos Requeridos

### 8.1 Recursos Computacionales

#### Track A (Modelos Tradicionales):
- **CPU:** Intel i7/AMD Ryzen 7 o superior
- **RAM:** 16GB mínimo, 32GB recomendado
- **Almacenamiento:** SSD 500GB para datasets procesados
- **Tiempo de procesamiento:** 2-4 horas por experimento

#### Track B (Deep Learning):
- **GPU:** NVIDIA RTX 3080/4080 o superior (12GB+ VRAM)
- **CPU:** Intel i9/AMD Ryzen 9 o superior  
- **RAM:** 32GB mínimo, 64GB recomendado
- **Almacenamiento:** SSD 1TB para checkpoints de modelos
- **Tiempo de procesamiento:** 8-24 horas por modelo LSTM, 24-72 horas para Transformers

### 8.2 Recursos Humanos

- **Científico de Datos Senior:** 0.8 FTE (tiempo completo equivalente)
- **Ingeniero de ML:** 0.5 FTE para implementación de producción
- **Especialista en Dominio:** 0.2 FTE para validación técnica
- **Total:** 1.5 FTE durante 4 meses

### 8.3 Software y Licencias

- **Python 3.8+** con librerías open source
- **TensorFlow/PyTorch** para Deep Learning
- **Weights & Biases** para experiment tracking ($50/mes)
- **MLflow** para model versioning (open source)

---

## 9. Cronograma Detallado

### 9.1 Track A - Optimización Tradicional

| Semana | Actividades | Entregables | Criterios de Éxito |
|---------|-------------|-------------|-------------------|
| 1 | Hiperparámetros + Umbral | Modelo RF optimizado | F1 > 0.30 |
| 2 | SMOTE + Validación temporal | Pipeline rebalanceado | F1 > 0.40 |
| 3 | Feature Engineering | Características avanzadas | F1 > 0.50 |
| 4 | Ensemble + Validación | Modelo final Track A | F1 > 0.55, Precision > 0.25 |

### 9.2 Track B - Deep Learning

| Mes | Actividades | Entregables | Criterios de Éxito |
|-----|-------------|-------------|-------------------|
| 1 | Arquitectura LSTM | Modelo base LSTM | F1 > 0.45 |
| 2 | Optimización LSTM | LSTM optimizado | F1 > 0.60 |
| 3 | Transformer | Modelo Transformer | F1 > 0.65 |
| 4 | CNN-LSTM híbrido | Arquitectura final | F1 > 0.70, Precision > 0.35 |

---

## 10. Métricas de Éxito y KPIs

### 10.1 Métricas Técnicas

| Métrica | Baseline Actual | Objetivo Track A | Objetivo Track B | Benchmark Industrial |
|---------|----------------|------------------|------------------|---------------------|
| F1-Score | 0.145 | 0.55+ | 0.70+ | 0.60-0.80 |
| Precision | 0.078 | 0.25+ | 0.35+ | 0.30-0.50 |
| Recall | 1.000 | 0.85+ | 0.90+ | 0.85-0.95 |
| AUC-ROC | 0.877 | 0.92+ | 0.94+ | 0.85-0.95 |
| Latencia | N/A | <1s | <5s | <10s |

### 10.2 Métricas Operacionales

- **Reducción de falsas alarmas:** >70% vs baseline
- **Costo operacional:** <$1M por período de evaluación
- **Disponibilidad del equipo:** Mejora proyectada 15-25%
- **ROI del sistema:** Positivo dentro de 12 meses

---

## 11. Conclusiones y Recomendaciones

### 11.1 Evaluación del Estado Actual

Los modelos base desarrollados demuestran la **viabilidad técnica** del enfoque de mantenimiento predictivo basado en Machine Learning. Random Forest muestra particular promesa con:
- Recall perfecto (100%) - crítico para seguridad
- AUC-ROC excelente (0.877) - capacidad discriminativa sólida
- Estrategia conservadora apropiada para equipos críticos

Sin embargo, la **precisión insuficiente (7.8%)** impide el despliegue inmediato en producción.

### 11.2 Recomendaciones Estratégicas

#### 11.2.1 Implementación Inmediata (Track A)
1. **Iniciar con optimizaciones de modelos tradicionales** por su rapidez de implementación y menor riesgo técnico
2. **Establecer pipeline de validación continua** con métricas operacionales
3. **Desarrollar sistema de alertas graduales** que incorpore probabilidades de falla

#### 11.2.2 Investigación Paralela (Track B)  
1. **Explorar arquitecturas LSTM** para captura de patrones temporales complejos
2. **Investigar Transformers adaptados** para series temporales industriales
3. **Desarrollar técnicas de interpretabilidad** específicas para el dominio

#### 11.2.3 Validación Industrial
1. **Piloto controlado** con Track A optimizado (6-8 semanas)
2. **Validación con expertos de dominio** para calibración de alertas
3. **Monitoreo de concept drift** en producción

### 11.3 Contribuciones Científicas Esperadas

1. **Marco metodológico** para ML en mantenimiento predictivo de equipos críticos
2. **Análisis comparativo** entre arquitecturas tradicionales y Deep Learning
3. **Técnicas de interpretabilidad** para modelos de series temporales industriales
4. **Metodología de evaluación** considerando costos operacionales reales

### 11.4 Impacto Proyectado

#### Impacto Técnico:
- Sistema de mantenimiento predictivo con F1-Score >0.55 (Track A) o >0.70 (Track B)
- Reducción >70% en falsas alarmas respecto a baseline
- Framework replicable para otros equipos rotativos

#### Impacto Operacional:
- Ahorro estimado $3-15 millones anuales por equipo
- Mejora 15-25% en disponibilidad operacional
- Reducción significativa en riesgos de seguridad

#### Impacto Académico:
- Contribución a literatura de ML industrial
- Metodología de validación rigurosa para sistemas críticos
- Framework de análisis costo-beneficio para modelos de ML

---

## Referencias

1. Carvalho, T. P., et al. (2019). "A systematic literature review of machine learning methods applied to predictive maintenance." *Computers & Industrial Engineering*, 137, 106024.

2. Lei, Y., et al. (2020). "Machinery health prognostics: A systematic review from data acquisition to RUL prediction." *Mechanical Systems and Signal Processing*, 104, 799-834.

3. Zhang, W., et al. (2019). "Deep learning for smart manufacturing: Methods and applications." *Journal of Manufacturing Systems*, 48, 144-156.

4. Mobley, R. K. (2002). *An introduction to predictive maintenance*. Elsevier.

5. Ahmad, W., et al. (2022). "A reliable technique for remaining useful life estimation of rolling element bearings using dynamic regression models." *Reliability Engineering & System Safety*, 184, 67-76.

---

**Anexo A:** Código completo de implementación  
**Anexo B:** Resultados detallados de experimentos  
**Anexo C:** Análisis de sensibilidad de hiperparámetros  

---

*Este documento constituye el plan técnico detallado para la optimización de modelos de Machine Learning en el proyecto de mantenimiento predictivo. La implementación seguirá la metodología propuesta con validación rigurosa en cada etapa del desarrollo.*