def bubble_sort(values, indices, reverse=True):
    """
    Bubble sort ile sıralama yapar, indisleri takip eder.
    """
    n = len(values)
    for i in range(n):
        for j in range(0, n - i - 1):
            a = values[j]
            b = values[j + 1]
            if reverse and a < b:
                values[j], values[j + 1] = b, a
                indices[j], indices[j + 1] = indices[j + 1], indices[j]
    return values, indices

def sort_column(results, column):
    """
    Sütunu bubble sort ile büyükten küçüğe sıralar, /sıra_no ekler.
    L/S/T ve S/S/T için momentum bazlı sıralama yapar.
    """
    symbol_map = {res['symbol']: idx + 1 for idx, res in enumerate(results)}
    n = len(results)
    indices = list(range(n))
    
    if column in ['L/S/T', 'S/S/T']:
        values = []
        for item in results:
            signal = item[column]
            if signal == '-':
                values.append(float('-inf'))
            else:
                momentum = float(signal.split('/')[0])
                values.append(momentum)
        
        sorted_values, sorted_indices = bubble_sort(values, indices)
        
        sorted_results = []
        for idx in sorted_indices:
            signal = results[idx][column]
            if signal == '-':
                sorted_results.append('-')
            else:
                momentum, mum_sayisi = signal.split('/')
                sira_no = symbol_map[results[idx]['symbol']]
                sorted_results.append(f"{float(momentum):.2f}/{mum_sayisi}/{sira_no}")
        
        return sorted_results
    
    values = [float(item[column]) if item[column] not in [None, ''] else float('-inf') for item in results]
    
    sorted_values, sorted_indices = bubble_sort(values, indices)
    
    sorted_results = [
        f"{sorted_values[i]:.2f}/{symbol_map[results[idx]['symbol']]}"
        for i, idx in enumerate(sorted_indices)
        if sorted_values[i] != float('-inf')
    ]
    
    return sorted_results

def sort_all_columns(results):
    """
    Tüm sütunları sıralar ve tabloyu döndürür.
    """
    columns = ['Score', 'Fiyat', 'Volume', 'Volatil', 'Moment', 'RSI', 'Ratio', 'L/S/T', 'S/S/T']
    sorted_table = [{'symbol': res['symbol']} for res in results]
    
    for col in columns:
        sorted_values = sort_column(results, col)
        for idx, value in enumerate(sorted_values):
            sorted_table[idx][col] = value
    
    return sorted_table