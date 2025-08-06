"""
Función COMPLETA para guardar datasets - Incluye TODOS los metadatos
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
import sys
import subprocess
import gc

warnings.filterwarnings('ignore')

def guardar_dataset_final(df, ruta_destino, nombre_base='clean_timeseries_data',
                         archivos_procesados=None, archivos_fallidos=None,
                         valores_interpolados=0, valores_clipped=0):
    """
    Guarda dataset en CSV (siempre) y Parquet (usando subprocess para evitar conflictos)
    Incluye todos los archivos de metadatos
    """
    ruta_destino = Path(ruta_destino)
    ruta_destino.mkdir(parents=True, exist_ok=True)
    
    print(f"💾 Guardando dataset final...")
    print(f"   📊 Dimensiones: {df.shape[0]:,} filas × {df.shape[1]} columnas")
    
    resultados = {}
    
    # ========== CSV (SIEMPRE) ==========
    try:
        archivo_csv = ruta_destino / f"{nombre_base}.csv"
        print(f"   📄 Guardando CSV...")
        df.to_csv(archivo_csv, index=True, encoding='utf-8')
        tamaño_mb = archivo_csv.stat().st_size / (1024 * 1024)
        print(f"      ✅ CSV: {tamaño_mb:.1f} MB")
        resultados['csv'] = {'exito': True, 'tamaño_mb': tamaño_mb}
    except Exception as e:
        print(f"      ❌ CSV Error: {str(e)}")
        return {'csv': {'exito': False, 'error': str(e)}}
    
    # ========== PARQUET (SUBPROCESS AISLADO) ==========
    archivo_parquet = ruta_destino / f"{nombre_base}.parquet"
    print(f"   📦 Guardando Parquet (subprocess aislado)...")
    
    try:
        # CSV temporal
        temp_csv = ruta_destino / 'temp_for_parquet.csv'
        df.to_csv(temp_csv, index=True)
        
        # Script aislado
        script_content = f'''
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

try:
    df = pd.read_csv("{temp_csv}", index_col=0, parse_dates=True)
    
    # Limpiar para Parquet
    if hasattr(df.index, 'hasnans') and df.index.hasnans:
        df = df[df.index.notna()]
    if str(type(df.index)) == "<class 'pandas.core.indexes.datetimes.DatetimeIndex'>":
        df = df.reset_index()
    
    df.to_parquet("{archivo_parquet}", engine='pyarrow', index=False, compression='snappy')
    print("PARQUET_SUCCESS")
    
except Exception as e:
    print(f"PARQUET_ERROR: {{e}}")
'''
        
        # Ejecutar subprocess
        result = subprocess.run([sys.executable, '-c', script_content], 
                              capture_output=True, text=True, timeout=120)
        
        # Limpiar
        if temp_csv.exists():
            temp_csv.unlink()
        
        # Verificar
        if result.returncode == 0 and "PARQUET_SUCCESS" in result.stdout:
            tamaño_mb = archivo_parquet.stat().st_size / (1024 * 1024)
            print(f"      ✅ Parquet: {tamaño_mb:.1f} MB")
            resultados['parquet'] = {'exito': True, 'tamaño_mb': tamaño_mb}
        else:
            print(f"      ❌ Parquet: Subprocess failed")
            resultados['parquet'] = {'exito': False, 'error': 'subprocess_failed'}
            
    except Exception as e:
        print(f"      ❌ Parquet Exception: {str(e)}")
        resultados['parquet'] = {'exito': False, 'error': str(e)}
    
    # ========== METADATOS COMPLETOS ==========
    try:
        print(f"   📄 Generando archivos de metadatos...")
        
        # 1. Metadatos principales
        archivo_meta = ruta_destino / 'preprocessing_metadata.txt'
        with open(archivo_meta, 'w', encoding='utf-8') as f:
            f.write(f"METADATOS DEL PREPROCESAMIENTO\n")
            f.write(f"=" * 40 + "\n\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Dimensiones: {df.shape[0]:,} × {df.shape[1]}\n")
            f.write(f"Archivos procesados: {len(archivos_procesados or [])}\n")
            f.write(f"Archivos fallidos: {len(archivos_fallidos or [])}\n")
            f.write(f"Valores interpolados: {valores_interpolados:,}\n")
            f.write(f"Valores clipped: {valores_clipped:,}\n")
            f.write(f"Índice temporal: {'Sí' if hasattr(df.index, 'tz') or 'datetime' in str(type(df.index)).lower() else 'No'}\n")
            f.write(f"\nFormatos guardados:\n")
            for formato, resultado in resultados.items():
                status = "✅" if resultado['exito'] else "❌" 
                f.write(f"  {formato.upper()}: {status}\n")
            
            if archivos_procesados:
                f.write(f"\nArchivos procesados exitosamente:\n")
                for archivo in archivos_procesados:
                    f.write(f"  - {archivo}\n")
            
            if archivos_fallidos:
                f.write(f"\nArchivos que fallaron:\n")
                for archivo in archivos_fallidos:
                    f.write(f"  - {archivo}\n")
        
        # 2. Resumen de columnas
        archivo_columnas = ruta_destino / 'column_summary.csv'
        columnas_info = pd.DataFrame({
            'columna': df.columns,
            'tipo_datos': [str(dtype) for dtype in df.dtypes],
            'valores_no_nulos': df.count(),
            'valores_nulos': df.isnull().sum(),
            'porcentaje_completitud': ((df.count() / len(df)) * 100).round(2)
        })
        
        # Agregar estadísticas para columnas numéricas
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        columnas_info['es_numerica'] = columnas_info['columna'].isin(numeric_cols)
        
        # Estadísticas básicas para columnas numéricas
        columnas_info['min_valor'] = np.nan
        columnas_info['max_valor'] = np.nan
        columnas_info['media'] = np.nan
        
        for col in numeric_cols:
            if not df[col].empty and df[col].notna().any():
                idx = columnas_info['columna'] == col
                try:
                    columnas_info.loc[idx, 'min_valor'] = df[col].min()
                    columnas_info.loc[idx, 'max_valor'] = df[col].max()
                    columnas_info.loc[idx, 'media'] = df[col].mean()
                except:
                    pass  # Ignorar errores en estadísticas
        
        columnas_info.to_csv(archivo_columnas, index=False, encoding='utf-8')
        
        # 3. Estadísticas descriptivas básicas (solo numéricas)
        if len(numeric_cols) > 0:
            archivo_stats = ruta_destino / 'basic_statistics.csv'
            stats_desc = df[numeric_cols].describe()
            stats_desc.to_csv(archivo_stats, encoding='utf-8')
        
        print(f"      ✅ preprocessing_metadata.txt")
        print(f"      ✅ column_summary.csv")
        if len(numeric_cols) > 0:
            print(f"      ✅ basic_statistics.csv")
        
    except Exception as e:
        print(f"      ⚠️  Error en metadatos: {str(e)}")
        pass
    
    # ========== RESUMEN FINAL ==========
    print(f"\n📋 Resumen del guardado:")
    exitosos = sum(1 for r in resultados.values() if r['exito'])
    for formato, resultado in resultados.items():
        if resultado['exito']:
            print(f"   ✅ {formato.upper()}: {resultado['tamaño_mb']:.1f} MB")
        else:
            print(f"   ❌ {formato.upper()}: Error")
    
    print(f"\n📁 Archivos generados en data/processed/:")
    print(f"   🗃️  {nombre_base}.csv - Dataset principal")
    if resultados.get('parquet', {}).get('exito'):
        print(f"   📦 {nombre_base}.parquet - Dataset comprimido")
    print(f"   📄 preprocessing_metadata.txt - Metadatos del procesamiento")
    print(f"   📊 column_summary.csv - Resumen de columnas")
    if len(df.select_dtypes(include=[np.number]).columns) > 0:
        print(f"   📈 basic_statistics.csv - Estadísticas descriptivas")
    
    print(f"✅ Guardado completado: {exitosos}/{len(resultados)} formatos exitosos")
    print(f"➡️  Listo para la siguiente fase: Feature Engineering (03_feature_engineering.ipynb)")
    
    return resultados