import numpy as np
import matplotlib.pyplot as plt

# TAREA 1: GENERACIÓN DE DATOS Y GRÁFICA 1

# Genera las muestras en el dominio x in [-2, 2] para la función
# objetivo y = x^2 + sin(5x) + epsilon.
def generar_dataset(n_puntos=350, sigma_ruido=0.18, seed=101):
    np.random.seed(seed)
    x = np.linspace(-2.0, 2.0, n_puntos).reshape(-1, 1)
    y_teorica = x**2 + np.sin(5.0 * x)
    epsilon = np.random.normal(0.0, sigma_ruido, size=x.shape)
    y_ruidosa = y_teorica + epsilon
    return x, y_ruidosa, y_teorica


x, y, y_teorica = generar_dataset(n_puntos=350, sigma_ruido=0.18, seed=101)

# Gráfica 1: Datos de entrenamiento vs Función teórica
plt.figure(figsize=(9, 5))
plt.scatter(x, y, color='steelblue', alpha=0.5, s=18, label='Datos de entrenamiento (con ruido)')
plt.plot(x, y_teorica, color='crimson', linewidth=2.2, label='Función objetivo $y = x^2 + \\sin(5x)$')
plt.title('Generación de Datos para el Aproximador No Lineal')
plt.xlabel('Entrada ($x$)')
plt.ylabel('Salida ($y$)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('fig1_datos.png', dpi=300)
plt.show()


# TAREA 2: PROGRAMACIÓN DEL ALGORITMO (FORWARD & BACKWARD PASS)

# Función de activación ReLU: max(0, z)
def relu(z):
    return np.maximum(0.0, z)


# Derivada de ReLU: 1 si z > 0, 0 en caso contrario
def d_relu(z):
    return np.where(z > 0.0, 1.0, 0.0)


# Inicializa los pesos (w1, w2) y los sesgos (b1, b2)
def inicializar_parametros(H, seed):
    np.random.seed(seed)
    params = {
        'w1': np.random.randn(1, H) * np.sqrt(2.0 / 1.0),
        'b1': np.random.uniform(-1.8, 1.8, size=(1, H)),
        'w2': np.random.randn(H, 1) * np.sqrt(2.0 / H),
        'b2': np.zeros((1, 1))
    }
    return params


# Forward Pass
def forward_pass(x, params):
    z1 = np.dot(x, params['w1']) + params['b1']
    a1 = relu(z1)
    y_hat = np.dot(a1, params['w2']) + params['b2']
    cache = {'x': x, 'z1': z1, 'a1': a1, 'y_hat': y_hat}
    return y_hat, cache


# Backpropagation
def backward_pass(y, params, cache):
    n = len(y)
    x, z1, a1, y_hat = cache['x'], cache['z1'], cache['a1'], cache['y_hat']

    # Gradientes de la capa de salida (Ecuación 3)
    delta_out = (1.0 / n) * (y_hat - y)
    dw2 = np.dot(a1.T, delta_out)
    db2 = np.sum(delta_out, axis=0, keepdims=True)

    # Gradientes de la capa oculta (Ecuaciones 4 y 5)
    delta_hidden = np.dot(delta_out, params['w2'].T) * d_relu(z1)
    dw1 = np.dot(x.T, delta_hidden)
    db1 = np.sum(delta_hidden, axis=0, keepdims=True)

    grads = {'dw1': dw1, 'db1': db1, 'dw2': dw2, 'db2': db2}
    return grads


# Bucle principal de optimización por Descenso de Gradiente
def entrenar_modelo(x, y, H=28, alpha=0.04, epocas=7000, seed=42):
    n = len(x)
    params = inicializar_parametros(H, seed)
    historial_costo = []

    for epoca in range(epocas):
        # 1. Propagación hacia adelante
        y_hat, cache = forward_pass(x, params)

        # Evaluación de la función de costo J
        residuo = y_hat - y
        costo = (1.0 / (2.0 * n)) * np.sum(residuo**2)
        historial_costo.append(costo)

        # Salvaguarda contra divergencia numérica (Overflow con alpha=0.5)
        if np.isnan(costo) or np.isinf(costo) or costo > 1e8:
            historial_costo.extend([np.nan] * (epocas - epoca - 1))
            break

        # 2. Retropropagación del error
        grads = backward_pass(y, params, cache)

        # 3. Actualización de parámetros
        params['w2'] -= alpha * grads['dw2']
        params['b2'] -= alpha * grads['db2']
        params['w1'] -= alpha * grads['dw1']
        params['b1'] -= alpha * grads['db1']

    y_final, _ = forward_pass(x, params)
    return y_final, historial_costo

# TAREA 3: ANÁLISIS DE MÍNIMOS LOCALES Y GRÁFICA 2

semillas_t3 = [12, 45, 108, 314, 777]
colores_t3 = ['red', 'blue', 'green', 'purple', 'orange']

# Gráfica 2: Superficie de respuesta para 5 semillas diferentes
plt.figure(figsize=(10, 5.5))
plt.scatter(x, y, color='lightgray', alpha=0.5, s=18, label='Datos ruidosos')
plt.plot(x, y_teorica, 'k--', linewidth=1.8, label='Función teórica')

for sem, col in zip(semillas_t3, colores_t3):
    y_pred, hist_j = entrenar_modelo(x, y, H=28, alpha=0.04, epocas=7000, seed=sem)
    plt.plot(x, y_pred, color=col, linewidth=2, label=f'Semilla {sem} ($J={hist_j[-1]:.4f}$)')

plt.title('Análisis de Mínimos Locales (5 Semillas Aleatorias)')
plt.xlabel('Entrada ($x$)')
plt.ylabel('Salida ($\\hat{y}$)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('fig2_minimos_locales.png', dpi=300)
plt.show()


# TAREA 4: IMPACTO DE LA TASA DE APRENDIZAJE Y GRÁFICA 3

tasas_alpha = [0.001, 0.01, 0.5]
colores_t4 = ['darkorange', 'seagreen', 'crimson']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

# Panel izquierdo: Superficie de respuesta comparada con los datos reales
ax1.scatter(x, y, color='lightgray', alpha=0.5, s=18, label='Datos ruidosos')
ax1.plot(x, y_teorica, 'k--', linewidth=1.8, label='Función teórica')

for alpha_val, col in zip(tasas_alpha, colores_t4):
    y_pred_a, hist_a = entrenar_modelo(x, y, H=28, alpha=alpha_val, epocas=7000, seed=42)

    # Verificar si el modelo convergió o divergió para graficar la curva
    ultimo_costo = hist_a[-1]
    if not np.isnan(ultimo_costo) and ultimo_costo < 100:
        ax1.plot(x, y_pred_a, color=col, linewidth=2.2,
                 label=f'$\\alpha = {alpha_val}$ ($J={ultimo_costo:.4f}$)')
    else:
        ax1.plot([], [], color=col, linewidth=2.2,
                 label=f'$\\alpha = {alpha_val}$ (Diverge / Overflow)')

    # Panel derecho: Evolución de la función de costo J por época
    ax2.plot(hist_a, color=col, linewidth=2, label=f'$\\alpha = {alpha_val}$')

ax1.set_title('Predicción según Tasa de Aprendizaje ($\\alpha$)')
ax1.set_xlabel('Entrada ($x$)')
ax1.set_ylabel('Salida ($\\hat{y}$)')
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.6)

ax2.set_title('Convergencia del Costo ($J$) vs. Épocas')
ax2.set_xlabel('Épocas')
ax2.set_ylabel('Costo $J$ (Escala Log)')
ax2.set_yscale('log')
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('fig3_tasas_aprendizaje.png', dpi=300)
plt.show()


# EXTRA 1: OPTIMIZACIÓN POBLACIONAL ESTÁNDAR (GLOBAL)

def entrenar_modelo_poblacional(x, y, H=28, alpha=0.04, total_epocas=7000, n_poblacion=5, seed_base=42):
    n = len(x)
    np.random.seed(seed_base)

    poblacion = [inicializar_parametros(H, seed=seed_base + i * 10) for i in range(n_poblacion)]
    historial_costo_lider = []
    bloque_epocas = 500
    iteraciones_bloque = total_epocas // bloque_epocas

    for bloque in range(iteraciones_bloque):
        costos_actuales = []

        for i, params in enumerate(poblacion):
            for _ in range(bloque_epocas):
                y_hat, cache = forward_pass(x, params)
                residuo = y_hat - y
                costo = (1.0 / (2.0 * n)) * np.sum(residuo**2)

                if np.isnan(costo) or np.isinf(costo):
                    costo = 1e9
                    break

                grads = backward_pass(y, params, cache)
                params['w2'] -= alpha * grads['dw2']
                params['b2'] -= alpha * grads['db2']
                params['w1'] -= alpha * grads['dw1']
                params['b1'] -= alpha * grads['db1']

            y_hat, _ = forward_pass(x, params)
            costo_final = (1.0 / (2.0 * n)) * np.sum((y_hat - y)**2)
            costos_actuales.append(costo_final)

        mejor_idx = np.argmin(costos_actuales)
        mejor_params = {k: poblacion[mejor_idx][k].copy() for k in ['w1', 'b1', 'w2', 'b2']}
        historial_costo_lider.append(costos_actuales[mejor_idx])

        # Evitar mutar en el último bloque para no alterar la solución final
        if bloque < iteraciones_bloque - 1:
            for i in range(n_poblacion):
                if i != mejor_idx:
                    for k in ['w1', 'b1', 'w2', 'b2']:
                        poblacion[i][k] = mejor_params[k].copy()
                        poblacion[i][k] += np.random.normal(0, 0.02, size=poblacion[i][k].shape)

    y_final, _ = forward_pass(x, mejor_params)
    return y_final, historial_costo_lider


# EXTRA 2: OPTIMIZACIÓN POBLACIONAL POR ZONAS Y RESCATE DE BORDES (ESTABILIZADO)

def inicializar_parametros_cobertura(H, seed):
    np.random.seed(seed)
    direcciones = np.random.choice([-1.0, 1.0], size=(1, H))
    # Magnitudes moderadas para evitar explosión de gradientes en los extremos
    magnitudes = np.random.uniform(1.2, 2.4, size=(1, H))
    w1 = direcciones * magnitudes

    # Repartir las bisagras uniformemente entre -1.90 y 1.90
    bisagras = np.linspace(-1.90, 1.90, H).reshape(1, H) + np.random.normal(0, 0.03, size=(1, H))
    bisagras = np.clip(bisagras, -1.95, 1.95)

    # Despeje geométrico: como z1 = w1*x + b1 = 0 en la bisagra -> b1 = -w1 * bisagra
    b1 = -w1 * bisagras
    w2 = np.random.randn(H, 1) * np.sqrt(1.0 / H)
    b2 = np.zeros((1, 1))
    return {'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2}


def entrenar_modelo_poblacional_zonas(x, y, H=28, alpha=0.025, total_epocas=10000, n_poblacion=6, seed_base=42):
    n = len(x)
    np.random.seed(seed_base)
    x_flat = x.ravel()

    # División del dominio [-2, 2] en 3 zonas: Izquierda, Centro y Derecha
    zonas_limites = [(-2.0, -0.8), (-0.8, 0.8), (0.8, 2.0)]
    mascaras_zonas = [(x_flat >= z_min) & (x_flat <= z_max) for z_min, z_max in zonas_limites]

    # Ponderación espacial suave para dar mayor sensibilidad a los bordes en el arranque
    pesos_bordes = 1.0 + 0.45 * (np.abs(x) / 2.0)**2

    poblacion = [inicializar_parametros_cobertura(H, seed=seed_base + i * 19) for i in range(n_poblacion)]
    historial_costo_lider = []
    bloque_epocas = 500
    iteraciones_bloque = total_epocas // bloque_epocas

    for bloque in range(iteraciones_bloque):
        for i, params in enumerate(poblacion):
            for _ in range(bloque_epocas):
                y_hat, cache = forward_pass(x, params)
                residuo = y_hat - y
                costo_iter = (1.0 / (2.0 * n)) * np.sum(residuo**2)

                # Salvaguarda contra divergencia numérica (Overflow / NaN)
                if np.isnan(costo_iter) or np.isinf(costo_iter) or costo_iter > 1e6:
                    break

                # En la primera mitad del entrenamiento se refuerza el gradiente en los bordes;
                # en la segunda mitad se optimiza el costo estándar exacto.
                if bloque < iteraciones_bloque // 2:
                    delta_out = (1.0 / n) * residuo * pesos_bordes
                    dw2 = np.dot(cache['a1'].T, delta_out)
                    db2 = np.sum(delta_out, axis=0, keepdims=True)
                    delta_hidden = np.dot(delta_out, params['w2'].T) * d_relu(cache['z1'])
                    dw1 = np.dot(x.T, delta_hidden)
                    db1 = np.sum(delta_hidden, axis=0, keepdims=True)
                    grads = {'dw1': dw1, 'db1': db1, 'dw2': dw2, 'db2': db2}
                else:
                    grads = backward_pass(y, params, cache)

                # Gradient Clipping: limita gradientes extremos para evitar explosión numérica
                for k in grads:
                    grads[k] = np.clip(grads[k], -5.0, 5.0)

                params['w2'] -= alpha * grads['dw2']
                params['b2'] -= alpha * grads['db2']
                params['w1'] -= alpha * grads['dw1']
                params['b1'] -= alpha * grads['db1']

        # Evaluar desempeño global y por zonas verificando que no existan NaNs
        costos_globales = []
        costos_zonas = np.full((n_poblacion, len(zonas_limites)), 1e9)

        for i, params in enumerate(poblacion):
            y_hat, _ = forward_pass(x, params)
            residuo_cuad = (y_hat - y)**2
            c_glob = (1.0 / (2.0 * n)) * np.sum(residuo_cuad)

            if np.isnan(c_glob) or np.isinf(c_glob):
                costos_globales.append(1e9)
            else:
                costos_globales.append(c_glob)
                for z_idx, mask in enumerate(mascaras_zonas):
                    costos_zonas[i, z_idx] = np.mean(residuo_cuad[mask])

        mejor_idx = int(np.argmin(costos_globales))
        campeones_zona = [int(np.argmin(costos_zonas[:, z_idx])) for z_idx in range(len(zonas_limites))]
        mejor_params = {k: poblacion[mejor_idx][k].copy() for k in ['w1', 'b1', 'w2', 'b2']}
        historial_costo_lider.append(costos_globales[mejor_idx])

        # En los últimos 2 bloques dejamos que el líder converja sin mutaciones
        if bloque >= iteraciones_bloque - 2:
            continue

        for i in range(n_poblacion):
            if i == mejor_idx:
                continue  # Elitismo: el mejor individuo global no se altera

            clon = {k: mejor_params[k].copy() for k in ['w1', 'b1', 'w2', 'b2']}

            # Estrategia A (Individuo 1): Cruce espacial heredando neuronas de los campeones zonales
            if i == (mejor_idx + 1) % n_poblacion:
                w1_safe_clon = np.where(np.abs(clon['w1'].ravel()) < 1e-5, 1e-5, clon['w1'].ravel())
                bisagras_clon = -clon['b1'].ravel() / w1_safe_clon

                for z_idx, (z_min, z_max) in enumerate(zonas_limites):
                    camp_params = poblacion[campeones_zona[z_idx]]
                    w1_safe_camp = np.where(np.abs(camp_params['w1'].ravel()) < 1e-5, 1e-5, camp_params['w1'].ravel())
                    bisagras_camp = -camp_params['b1'].ravel() / w1_safe_camp

                    idx_clon_z = np.where((bisagras_clon >= z_min) & (bisagras_clon <= z_max))[0]
                    idx_camp_z = np.where((bisagras_camp >= z_min) & (bisagras_camp <= z_max))[0]

                    k_intercambio = min(len(idx_clon_z), len(idx_camp_z))
                    if k_intercambio > 0:
                        c_dest = idx_clon_z[:k_intercambio]
                        c_orig = idx_camp_z[:k_intercambio]
                        clon['w1'][0, c_dest] = 0.5 * clon['w1'][0, c_dest] + 0.5 * camp_params['w1'][0, c_orig]
                        clon['b1'][0, c_dest] = 0.5 * clon['b1'][0, c_dest] + 0.5 * camp_params['b1'][0, c_orig]
                        clon['w2'][c_dest, 0] = 0.5 * clon['w2'][c_dest, 0] + 0.5 * camp_params['w2'][c_orig, 0]

            # Estrategia B (Resto de la población): Rescate de neuronas muertas hacia los bordes con error
            else:
                y_hat_lider, cache_lider = forward_pass(x, clon)
                activaciones = np.sum(cache_lider['a1'] > 0, axis=0)
                w1_safe = np.where(np.abs(clon['w1'].ravel()) < 1e-5, 1e-5, clon['w1'].ravel())
                bisagras = -clon['b1'].ravel() / w1_safe

                # Detectar neuronas inactivas o con bisagras fuera de [-2, 2]
                inactivas = np.where((activaciones < 8) | (np.abs(bisagras) > 1.98))[0]
                error_residual = np.nan_to_num(np.abs(y_hat_lider - y).ravel(), nan=1.0, posinf=1.0)
                suma_err = np.sum(error_residual**2)

                if suma_err > 1e-12:
                    prob_error = (error_residual**2) / suma_err
                else:
                    prob_error = np.ones_like(x_flat) / len(x_flat)

                for idx_n in inactivas:
                    # Ubicar la nueva bisagra exactamente donde el error residual es crítico
                    x_critico = np.random.choice(x_flat, p=prob_error)
                    nueva_pendiente = np.random.choice([-1.0, 1.0]) * np.random.uniform(1.5, 2.5)
                    clon['w1'][0, idx_n] = nueva_pendiente
                    clon['b1'][0, idx_n] = -nueva_pendiente * x_critico
                    # Peso de salida en 0 para no alterar la red bruscamente al insertar la bisagra
                    clon['w2'][idx_n, 0] = 0.0

                for k in ['w1', 'b1', 'w2']:
                    clon[k] += np.random.normal(0, 0.01, size=clon[k].shape)

            poblacion[i] = clon

    y_final, _ = forward_pass(x, mejor_params)
    return y_final, historial_costo_lider


# EJECUCIÓN Y GRÁFICAS SEPARADAS DE LOS MODELOS POBLACIONALES

# --- GRÁFICA 4: Modelo Poblacional Estándar (Muestra el problema en los bordes) ---
print("1. Entrenando Modelo Poblacional Estándar...")
y_pred_pob_normal, hist_pob_normal = entrenar_modelo_poblacional(
    x, y, H=28, alpha=0.04, total_epocas=7000, n_poblacion=5, seed_base=42
)

plt.figure(figsize=(10, 5.5))
plt.scatter(x, y, color='lightgray', alpha=0.5, s=18, label='Datos ruidosos')
plt.plot(x, y_teorica, 'k--', linewidth=1.8, label='Función teórica $y = x^2 + \\sin(5x)$')
plt.plot(x, y_pred_pob_normal, color='purple', linewidth=2.5,
         label=f'Optimización Poblacional Estándar ($J={hist_pob_normal[-1]:.5f}$)')

plt.title('Extra 1: Optimización Poblacional Global (Defecto de Linealidad en Bordes)')
plt.xlabel('Entrada ($x$)')
plt.ylabel('Salida ($\\hat{y}$)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('fig4_extra_poblacional_estandar.png', dpi=300)
plt.show()


# --- GRÁFICA 5: Modelo Poblacional por Zonas (Muestra la corrección en los bordes) ---
print("2. Entrenando Modelo Poblacional por Zonas (Mitigación de Bordes)...")
y_pred_pob_zonas, hist_pob_zonas = entrenar_modelo_poblacional_zonas(
    x, y, H=28, alpha=0.025, total_epocas=10000, n_poblacion=6, seed_base=42
)

plt.figure(figsize=(10, 5.5))
plt.scatter(x, y, color='lightgray', alpha=0.5, s=18, label='Datos ruidosos')
plt.plot(x, y_teorica, 'k--', linewidth=1.8, label='Función teórica $y = x^2 + \\sin(5x)$')
plt.plot(x, y_pred_pob_zonas, color='teal', linewidth=2.5,
         label=f'Optimización Poblacional por Zonas ($J={hist_pob_zonas[-1]:.5f}$)')

plt.title('Extra 2: Optimización Poblacional con Especialización Zonal y Rescate de Bordes')
plt.xlabel('Entrada ($x$)')
plt.ylabel('Salida ($\\hat{y}$)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('fig5_extra_poblacional_zonas.png', dpi=300)
plt.show()
