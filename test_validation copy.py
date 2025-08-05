#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de validación para verificar si las funciones de carga de Excel funcionan correctamente
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_sensor_file(file_path, max_rows=None):
    """
    Carga un archivo de sensores con manejo inteligente de estructura compleja de Excel
    """
    try:
        # Primero, cargar el archivo completo para analizar su estructura
        if file_path.suffix == '.xlsx':
            df_raw = pd.read_excel(file_path, header=None, engine='openpyxl')
        else:
            df_raw = pd.read_excel(file_path, header=None, engine='xlrd')
        
        # Buscar la fila donde comienzan los datos reales
        # Buscar filas que contengan 'COMPRESOR' o 'MOTOR' como indicadores
        header_row = None
        data_start_row = None
        
        for idx, row in df_raw.iterrows():
            row_str = ' '.join(str(cell) for cell in row if pd.notna(cell))
            if 'COMPRESOR' in row_str.upper() or 'MOTOR' in row_str.upper():
                header_row = idx
                data_start_row = idx + 1
                break
        
        # Si no encontramos indicadores específicos, buscar la primera fila con datos numéricos
        if header_row is None:
            for idx, row in df_raw.iterrows():
                # Verificar si la fila tiene datos numéricos
                numeric_count = sum(1 for cell in row if pd.notna(cell) and 
                                  str(cell).replace('.', '').replace('-', '').replace(':', '').isdigit())
                if numeric_count > 5:  # Si más de 5 celdas tienen datos numéricos
                    header_row = idx - 1
                    data_start_row = idx
                    break
        
        # Si aún no encontramos, usar una estrategia más agresiva
        if header_row is None:
            # Buscar la primera fila que no esté completamente vacía
            for idx, row in df_raw.iterrows():
                if not row.isna().all():
                    header_row = idx
                    data_start_row = idx + 1
                    break
        
        # Cargar el archivo con los parámetros correctos
        if file_path.suffix == '.xlsx':
            df = pd.read_excel(file_path, 
                             header=header_row if header_row is not None else 0,
                             skiprows=range(0, header_row) if header_row is not None else None,
                             nrows=max_rows,
                             engine='openpyxl')
        else:
            df = pd.read_excel(file_path, 
                             header=header_row if header_row is not None else 0,
                             skiprows=range(0, header_row) if header_row is not None else None,
                             nrows=max_rows,
                             engine='xlrd')
        
        # Limpiar nombres de columnas
        df.columns = [str(col).strip().replace('\n', ' ') if pd.notna(col) else f'Col_{i}' 
                     for i, col in enumerate(df.columns)]
        
        # Eliminar filas completamente vacías
        df = df.dropna(how='all')
        
        # Eliminar columnas completamente vacías
        df = df.dropna(axis=1, how='all')
        
        return {
            'dataframe': df,
            'filename': file_path.name,
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum() / (1024*1024),  # MB
            'success': True,
            'error': None,
            'header_row_found': header_row,
            'data_start_row': data_start_row
        }
    except Exception as e:
        return {
            'dataframe': None,
            'filename': file_path.name,
            'shape': (0, 0),
            'columns': [],
            'dtypes': {},
            'memory_usage': 0,
            'success': False,
            'error': str(e),
            'header_row_found': None,
            'data_start_row': None
        }

def test_single_file():
    """Probar la carga de un solo archivo"""
    print("🔍 PRUEBA DE CARGA DE ARCHIVO INDIVIDUAL")
    print("=" * 50)
    
    # Definir rutas
    data_path = Path('data/raw')
    sensor_files = list(data_path.glob('*.xls')) + list(data_path.glob('*.xlsx'))
    sensor_files = [f for f in sensor_files if 'Historial' not in f.name and f.suffix in ['.xls', '.xlsx']]
    
    if not sensor_files:
        print("❌ No se encontraron archivos de sensores")
        return
    
    # Probar con el primer archivo
    test_file = sensor_files[0]
    print(f"📁 Archivo de prueba: {test_file.name}")
    
    # Cargar con la función mejorada
    result = load_sensor_file(test_file, max_rows=100)
    
    if result['success']:
        df = result['dataframe']
        print(f"✅ Carga exitosa")
        print(f"📏 Dimensiones: {result['shape'][0]:,} filas × {result['shape'][1]} columnas")
        print(f"🔢 Fila de encabezado encontrada: {result['header_row_found']}")
        print(f"🔢 Fila de datos encontrada: {result['data_start_row']}")
        
        print(f"\n📋 Columnas encontradas:")
        for i, col in enumerate(result['columns'][:10], 1):
            print(f"   {i:2d}. {col}")
        
        if len(result['columns']) > 10:
            print(f"   ... y {len(result['columns']) - 10} columnas más")
        
        print(f"\n📊 Muestra de datos:")
        print(df.head(3))
        
        # Verificar si hay columnas "Unnamed"
        unnamed_cols = [col for col in df.columns if 'Unnamed' in str(col)]
        if unnamed_cols:
            print(f"\n⚠️ COLUMNAS PROBLEMÁTICAS ENCONTRADAS:")
            for col in unnamed_cols:
                print(f"   • {col}")
        else:
            print(f"\n✅ No se encontraron columnas 'Unnamed'")
            
    else:
        print(f"❌ Error: {result['error']}")

def test_raw_loading():
    """Probar carga directa sin procesamiento"""
    print("\n🔍 PRUEBA DE CARGA DIRECTA (SIN PROCESAMIENTO)")
    print("=" * 50)
    
    data_path = Path('data/raw')
    sensor_files = list(data_path.glob('*.xls')) + list(data_path.glob('*.xlsx'))
    sensor_files = [f for f in sensor_files if 'Historial' not in f.name and f.suffix in ['.xls', '.xlsx']]
    
    if not sensor_files:
        print("❌ No se encontraron archivos de sensores")
        return
    
    test_file = sensor_files[0]
    print(f"📁 Archivo de prueba: {test_file.name}")
    
    try:
        # Carga directa sin procesamiento
        if test_file.suffix == '.xlsx':
            df_raw = pd.read_excel(test_file, header=None, engine='openpyxl')
        else:
            df_raw = pd.read_excel(test_file, header=None, engine='xlrd')
        
        print(f"✅ Carga directa exitosa")
        print(f"📏 Dimensiones raw: {df_raw.shape[0]:,} filas × {df_raw.shape[1]} columnas")
        
        print(f"\n📋 Primeras 10 filas (raw):")
        print(df_raw.head(10))
        
        # Buscar filas con contenido relevante
        print(f"\n🔍 Buscando filas con contenido relevante:")
        for idx, row in df_raw.iterrows():
            row_str = ' '.join(str(cell) for cell in row if pd.notna(cell))
            if 'COMPRESOR' in row_str.upper() or 'MOTOR' in row_str.upper():
                print(f"   Fila {idx}: {row_str[:100]}...")
                break
        
    except Exception as e:
        print(f"❌ Error en carga directa: {e}")

if __name__ == "__main__":
    print("🚀 INICIANDO VALIDACIÓN DE FUNCIONES DE CARGA")
    print("=" * 60)
    
    test_single_file()
    test_raw_loading()
    
    print("\n✅ Validación completada") 