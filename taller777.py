import numpy as np
import matplotlib.pyplot as plt

# GENERACIÓN DE DATOS Y GRÁFICA 1

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


# PROGRAMACIÓN DEL ALGORITMO (FORWARD & BACKWARD PASS)

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


# ANÁLISIS DE MÍNIMOS LOCALES Y GRÁFICA 2

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

def entrenar_modelo_poblacional(x, y, H=28, alpha=0.04, total_epocas=7000, n_poblacion=5, seed_base=42):
    """
    Entrena una población de redes en paralelo. Periódicamente, las redes con
    peores costos heredan los pesos de la mejor red de la población (la líder),
    añadiendo una pequeña mutación aleatoria para escapar de mínimos locales.
    """
    n = len(x)
    np.random.seed(seed_base)

    # 1. Inicializar una población de redes (cada una con su propio set de parámetros)
    poblacion = []
    for i in range(n_poblacion):
        # Usamos semillas distintas para cada individuo
        params = inicializar_parametros(H, seed=seed_base + i * 10)
        poblacion.append(params)

    historial_costo_lider = []
    bloque_epocas = 500  # Cada cuántas épocas interactúan las redes
    iteraciones_bloque = total_epocas // bloque_epocas

    for bloque in range(iteraciones_bloque):
        costos_actuales = []

        # Entrenar cada red de la población por un bloque de épocas
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

            # Recalcular costo final del bloque para este individuo
            y_hat, _ = forward_pass(x, params)
            costo_final = (1.0 / (2.0 * n)) * np.sum((y_hat - y)**2)
            costos_actuales.append(costo_final)

        # Identificar al mejor individuo (el de menor costo en este bloque)
        mejor_idx = np.argmin(costos_actuales)
        mejor_params = poblacion[mejor_idx]
        historial_costo_lider.append(costos_actuales[mejor_idx])

        # Elitismo y Cruce (La idea que mencionaste):
        # Las redes rezagadas heredan los pesos del líder con una ligera mutación
        for i in range(n_poblacion):
            if i != mejor_idx:
                # Copiar los parámetros del líder
                for k in ['w1', 'b1', 'w2', 'b2']:
                    poblacion[i][k] = mejor_params[k].copy()
                    # Añadir mutación aleatoria gaussiana para explorar el vecindario
                    poblacion[i][k] += np.random.normal(0, 0.02, size=poblacion[i][k].shape)

    # Retornar la predicción de la mejor red al finalizar todas las épocas
    y_final, _ = forward_pass(x, mejor_params)
    return y_final, historial_costo_lider

# Ejecución y Gráfica del Extra Poblacional
print("Entrenando con el Algoritmo Poblacional Híbrido (Extra)...")
y_pred_poblacional, hist_poblacional = entrenar_modelo_poblacional(x, y, H=28, alpha=0.04, total_epocas=7000, n_poblacion=5, seed_base=42)

plt.figure(figsize=(10, 5.5))
plt.scatter(x, y, color='lightgray', alpha=0.5, s=18, label='Datos ruidosos')
plt.plot(x, y_teorica, 'k--', linewidth=1.8, label='Función teórica')
plt.plot(x, y_pred_poblacional, color='purple', linewidth=2.5, label=f'Optimización Poblacional (Híbrida) [Coste: {hist_poblacional[-1]:.5f}]')

plt.title('Extra: Superación de Mínimos Locales mediante Enfoque Poblacional')
plt.xlabel('Entrada ($x$)')
plt.ylabel('Salida ($\\hat{y}$)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('fig_extra_poblacional.png', dpi=300)
plt.show()
