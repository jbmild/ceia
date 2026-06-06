### **Reemplazo para la sección: 1) Diferencias entre `QDA` y `TensorizedQDA`**

#### **Punto 1: ¿Sobre qué paraleliza `TensorizedQDA`?**

El modelo `QDA` base implementa la predicción de una manera secuencial. Para una observación dada, representada por el vector $\mathbf{x}$, recorre cada una de las $k$ clases para calcular la función discriminante $\delta_j(\mathbf{x})$. Este proceso se repite para cada una de las $n$ observaciones en el conjunto de prueba.

`TensorizedQDA`, en cambio, aprovecha las capacidades de cómputo vectorial de NumPy para optimizar uno de estos procesos. En lugar de iterar sobre las $k$ clases, apila los parámetros de cada clase (las medias $\boldsymbol{\mu}_j$ y las inversas de las covarianzas $\mathbf{\Sigma}_j^{-1}$) en estructuras de datos de mayor dimensión, conocidas como tensores. Esto le permite calcular las $k$ funciones discriminantes para una única observación $\mathbf{x}$ de manera simultánea, a través de operaciones de broadcasting y multiplicación de matrices por lotes (batched matrix multiplication).

Sin embargo, es crucial notar que `TensorizedQDA` aún procesa las observaciones una por una. El método `predict` de la clase base, que contiene el bucle `for` sobre las $n$ observaciones, no se modifica. Por lo tanto, podemos afirmar que:

**`TensorizedQDA` paraleliza el cálculo sobre las $k$ clases, pero no sobre las $n$ observaciones a predecir.**

---

#### **Punto 2: Análisis de shapes y lógica de `TensorizedQDA`**

Para entender cómo `TensorizedQDA` logra el mismo resultado que `QDA` sin un bucle explícito sobre las clases, primero debemos definir la notación y los parámetros involucrados.

*   $k$: Número de clases.
*   $p$: Número de features (dimensiones).
*   $\boldsymbol{\mu}_j \in \mathbb{R}^{p \times 1}$: Vector de medias para la clase $j$.
*   $\mathbf{\Sigma}_j \in \mathbb{R}^{p \times p}$: Matriz de covarianza para la clase $j$.
*   $\mathbf{x} \in \mathbb{R}^{p \times 1}$: Una observación a clasificar.

En la etapa de `fit`, el modelo calcula las listas de estos parámetros. `TensorizedQDA` luego las convierte a tensores:

*   `self.means` es una lista de $k$ vectores de forma `(p, 1)`. Al apilarlos, `tensor_means` se convierte en un tensor $\boldsymbol{\mathcal{M}}$ de forma `(k, p, 1)`.
*   `self.inv_covs` es una lista de $k$ matrices de forma `(p, p)`. Al apilarlos, `tensor_inv_cov` se convierte en un tensor $\mathbf{\mathcal{S}}^{-1}$ de forma `(k, p, p)`.

Ahora, analicemos el método `_predict_log_conditionals` paso a paso para una observación $\mathbf{x}$:

1.  **Cálculo de las diferencias (centrado):**
    El primer paso es centrar la observación $\mathbf{x}$ con respecto a la media de cada clase. En lugar de hacerlo en un bucle, se aprovecha el broadcasting de NumPy. Restamos el tensor de medias $\boldsymbol{\mathcal{M}}$ (forma `(k, p, 1)`) del vector $\mathbf{x}$ (forma `(p, 1)`). NumPy "expande" $\mathbf{x}$ para que coincida con la forma de $\boldsymbol{\mathcal{M}}$, realizando $k$ restas en una sola operación. El resultado es un tensor que contiene todos los vectores de diferencia.

    $$
    \underset{(k, p, 1)}{\mathbf{\Delta}} = \underset{(p, 1)}{\mathbf{x}} - \underset{(k, p, 1)}{\boldsymbol{\mathcal{M}}}
    $$

    Donde la $j$-ésima "rebanada" (slice) de $\mathbf{\Delta}$ es el vector $\boldsymbol{\delta}_j = \mathbf{x} - \boldsymbol{\mu}_j$.

2.  **Cálculo de la forma cuadrática:**
    El siguiente paso es calcular la distancia de Mahalanobis al cuadrado para cada clase, que es la forma cuadrática $\boldsymbol{\delta}_j^T \mathbf{\Sigma}_j^{-1} \boldsymbol{\delta}_j$. NumPy realiza esta operación para las $k$ clases simultáneamente. El producto `@` entre tensores efectúa una multiplicación de matrices por lotes en los dos últimos ejes.

    $$
    \underset{(k, 1, 1)}{\mathbf{Q}} = \underset{(k, 1, p)}{\mathbf{\Delta}^T} \ @ \ \underset{(k, p, p)}{\mathbf{\mathcal{S}}^{-1}} \ @ \ \underset{(k, p, 1)}{\mathbf{\Delta}}
    $$

    El resultado $\mathbf{Q}$ es un tensor donde la $j$-ésima rebanada contiene el valor escalar de la forma cuadrática para la clase $j$. El método `.flatten()` lo convierte en un vector de forma `(k,)`.

3.  **Cálculo del término del determinante:**
    Finalmente, el término log-determinante se calcula de manera similar. La función `np.linalg.det` aplicada a un tensor de forma `(k, p, p)` devuelve un vector de forma `(k,)` que contiene el determinante de cada una de las $k$ matrices.

    $$
    \underset{(k,)}{\mathbf{d}} = \text{det}(\underset{(k, p, p)}{\mathbf{\mathcal{S}}^{-1}})
    $$

Al combinar estos componentes, el método calcula el vector completo de log-verosimilitudes condicionales, una para cada clase, sin necesidad de un bucle `for`.

---

### **Reemplazo para la sección: Optimización**

#### **Punto 4: Dónde aparece la matriz de $n \times n$**

La matriz de $n \times n$ emerge cuando intentamos extender la paralelización del cálculo de la forma cuadrática para que no solo abarque las $k$ clases, sino también las $n$ observaciones del conjunto de prueba simultáneamente.

Consideremos el cálculo para una sola clase $j$. Si en lugar de una única observación $\mathbf{x} \in \mathbb{R}^{p \times 1}$, queremos procesar el conjunto completo de $n$ observaciones, que representamos como una matriz $\mathbf{X} \in \mathbb{R}^{p \times n}$:

1.  **Centrado del conjunto de datos:**
    Primero, centramos cada columna (observación) de $\mathbf{X}$ restando el vector de medias $\boldsymbol{\mu}_j$. Esto se logra mediante broadcasting:

    $$
    \underset{(p, n)}{\mathbf{U}_j} = \underset{(p, n)}{\mathbf{X}} - \underset{(p, 1)}{\boldsymbol{\mu}_j}
    $$

2.  **Cálculo de la forma cuadrática matricial:**
    La forma cuadrática para el conjunto de datos completo se calcula como:

    $$
    \underset{(n, n)}{\mathbf{Q}_j} = \underset{(n, p)}{\mathbf{U}_j^T} \ \underset{(p, p)}{\mathbf{\Sigma}_j^{-1}} \ \underset{(p, n)}{\mathbf{U}_j}
    $$

Aquí es donde se materializa la matriz de $n \times n$. Cada elemento $(i, m)$ de esta matriz $\mathbf{Q}_j$ representa el producto cruzado $(\mathbf{x}_i - \boldsymbol{\mu}_j)^T \mathbf{\Sigma}_j^{-1} (\mathbf{x}_m - \boldsymbol{\mu}_j)$. Para la función discriminante, solo nos interesan los elementos de la diagonal, donde $i=m$, que corresponden a la distancia de Mahalanobis al cuadrado para cada observación. Construir la matriz completa es computacionalmente costoso y un derroche de memoria, ya que la mayor parte de la información calculada (los elementos fuera de la diagonal) se descarta.

---

#### **Punto 5: Demostración para "esquivar" la matriz de $n \times n$**

**Proposición:** Dada una matriz $\mathbf{A} \in \mathbb{R}^{n \times p}$ y una matriz $\mathbf{B} \in \mathbb{R}^{p \times n}$, la diagonal del producto matricial $\mathbf{C} = \mathbf{A}\mathbf{B}$ puede ser calculada eficientemente como la suma por filas del producto de Hadamard (elemento a elemento) entre $\mathbf{A}$ y $\mathbf{B}^T$.

$$
\text{diag}(\mathbf{A}\mathbf{B}) = \sum_{\text{cols}} (\mathbf{A} \odot \mathbf{B}^T)
$$

**Demostración:**

Por definición, el elemento $(i, j)$ de la matriz producto $\mathbf{C} = \mathbf{A}\mathbf{B}$ se calcula como:

$$
C_{ij} = (\mathbf{A}\mathbf{B})_{ij} = \sum_{k=1}^{p} A_{ik} B_{kj}
$$

Nos interesan los elementos de la diagonal de $\mathbf{C}$, que son aquellos donde $i=j$:

$$
C_{ii} = (\mathbf{A}\mathbf{B})_{ii} = \sum_{k=1}^{p} A_{ik} B_{ki}
$$

Ahora, consideremos la segunda parte de la proposición. La transpuesta de $\mathbf{B}$ es $\mathbf{B}^T \in \mathbb{R}^{n \times p}$, donde el elemento $(i, k)$ es $(B^T)_{ik} = B_{ki}$.

El producto de Hadamard (o elemento a elemento) de $\mathbf{A}$ y $\mathbf{B}^T$, que denotamos $\mathbf{A} \odot \mathbf{B}^T$, es una matriz $\mathbf{H} \in \mathbb{R}^{n \times p}$ cuyo elemento $(i, k)$ es:

$$
H_{ik} = (\mathbf{A} \odot \mathbf{B}^T)_{ik} = A_{ik} \cdot (B^T)_{ik} = A_{ik} \cdot B_{ki}
$$

Si ahora sumamos los elementos de cada fila de $\mathbf{H}$ (es decir, sumamos sobre el índice de las columnas, $k$), obtenemos un vector cuyos componentes $i$ son:

$$
\left( \sum_{\text{cols}} \mathbf{H} \right)_i = \sum_{k=1}^{p} H_{ik} = \sum_{k=1}^{p} A_{ik} \cdot B_{ki}
$$

Este resultado es idéntico al elemento $C_{ii}$ de la diagonal de $\mathbf{A}\mathbf{B}$. Por lo tanto, hemos demostrado la igualdad.

La ventaja computacional es inmensa: en lugar de realizar una multiplicación matricial con un costo de $O(n^2 p)$ y almacenar una matriz intermedia de $O(n^2)$, realizamos un producto de Hadamard y una suma con un costo y almacenamiento de solo $O(np)$.

---

### **Reemplazo para la sección: Diferencias entre implementaciones de `QDA_Chol`**

#### **Punto 8: Expresión de $A^{-1}$ en términos de Cholesky**

Dada una matriz simétrica y definida positiva $\mathbf{A}$, su descomposición de Cholesky es $\mathbf{A} = \mathbf{L}\mathbf{L}^T$, donde $\mathbf{L}$ es una matriz triangular inferior.

Para encontrar la inversa de $\mathbf{A}$, invertimos ambos lados de la ecuación:

$$
\mathbf{A}^{-1} = (\mathbf{L}\mathbf{L}^T)^{-1}
$$

Utilizando la propiedad de la inversa de un producto de matrices, $(XY)^{-1} = Y^{-1}X^{-1}$, obtenemos:

$$
\mathbf{A}^{-1} = (\mathbf{L}^T)^{-1} \mathbf{L}^{-1}
$$

Además, como la inversa de la transpuesta es la transpuesta de la inversa, $(\mathbf{L}^T)^{-1} = (\mathbf{L}^{-1})^T$, podemos reescribir la expresión como:

$$
\mathbf{A}^{-1} = (\mathbf{L}^{-1})^T \mathbf{L}^{-1}
$$

Esta formulación es extremadamente útil en la forma cuadrática de QDA. La expresión original $(\mathbf{x}-\boldsymbol{\mu})^T \mathbf{\Sigma}^{-1} (\mathbf{x}-\boldsymbol{\mu})$ se transforma. Sustituyendo $\mathbf{\Sigma}^{-1} = (\mathbf{L}^{-1})^T \mathbf{L}^{-1}$, tenemos:

$$
(\mathbf{x}-\boldsymbol{\mu})^T (\mathbf{L}^{-1})^T \mathbf{L}^{-1} (\mathbf{x}-\boldsymbol{\mu})
$$

Agrupando términos, esto es equivalente a:

$$
(\mathbf{L}^{-1}(\mathbf{x}-\boldsymbol{\mu}))^T (\mathbf{L}^{-1}(\mathbf{x}-\boldsymbol{\mu}))
$$

Si definimos un nuevo vector $\mathbf{y} = \mathbf{L}^{-1}(\mathbf{x}-\boldsymbol{\mu})$, la forma cuadrática se simplifica a $\mathbf{y}^T\mathbf{y}$, que no es más que la norma euclidiana al cuadrado de $\mathbf{y}$, $\|\mathbf{y}\|_2^2$. Esto reemplaza una multiplicación matriz-vector-matriz por una multiplicación matriz-vector seguida de un producto punto, lo cual es más eficiente y numéricamente más estable.

---

#### **Punto 9: Diferencias y lógica de `QDA_Chol1`**

La principal diferencia radica en cómo se calcula y utiliza la inversa de la matriz de covarianza.

*   **`QDA`:** Calcula la matriz de covarianza $\mathbf{\Sigma}_j$ y luego su inversa $\mathbf{\Sigma}_j^{-1}$ directamente usando una rutina de inversión de propósito general como `np.linalg.inv()`. Durante la predicción, calcula explícitamente la forma cuadrática $\boldsymbol{\delta}_j^T \mathbf{\Sigma}_j^{-1} \boldsymbol{\delta}_j$.
*   **`QDA_Chol1`:** Evita la inversión directa de $\mathbf{\Sigma}_j$. En su lugar, sigue un camino numéricamente más robusto:
    1.  Calcula $\mathbf{\Sigma}_j$.
    2.  Obtiene la descomposición de Cholesky: $\mathbf{\Sigma}_j = \mathbf{L}_j \mathbf{L}_j^T$.
    3.  Calcula y almacena la inversa de la matriz triangular inferior, $\mathbf{L}_j^{-1}$.

**Paso a paso de la predicción en `QDA_Chol1`:**

1.  **Centrado:** Se calcula el vector de diferencia $\boldsymbol{\delta}_j = \mathbf{x} - \boldsymbol{\mu}_j$.
2.  **Transformación:** Se calcula el vector transformado $\mathbf{y}_j = \mathbf{L}_j^{-1} \boldsymbol{\delta}_j$.
3.  **Cálculo de la norma:** La forma cuadrática se obtiene calculando la suma de los cuadrados de los elementos de $\mathbf{y}_j$, que es $\|\mathbf{y}_j\|_2^2 = \mathbf{y}_j^T \mathbf{y}_j$. Como demostramos anteriormente, esto es matemáticamente equivalente a la forma cuadrática original.
4.  **Cálculo del determinante:** El término log-determinante también se simplifica. El determinante de una matriz triangular es el producto de sus elementos diagonales. Por lo tanto:
    $$
    \frac{1}{2} \log |\mathbf{\Sigma}_j^{-1}| = \frac{1}{2} \log |(\mathbf{L}_j^{-1})^T \mathbf{L}_j^{-1}| = \log |\mathbf{L}_j^{-1}| = \log \left( \prod_{i=1}^{p} (L_j^{-1})_{ii} \right)
    $$
    Esto evita una llamada costosa a `np.linalg.det` y la reemplaza por un producto de escalares.

En resumen, `QDA_Chol1` reemplaza operaciones matriciales complejas por operaciones con matrices triangulares, que son más rápidas y estables.

---

#### **Punto 10: Diferencias entre `QDA_Chol1`, `QDA_Chol2` y `QDA_Chol3`**

Las tres variantes utilizan la descomposición de Cholesky, pero difieren en la estrategia para calcular el vector transformado $\mathbf{y}_j = \mathbf{L}_j^{-1} \boldsymbol{\delta}_j$.

*   **`QDA_Chol1`:** Sigue un enfoque de "calcular y guardar". Durante el `fit`, calcula explícitamente la matriz inversa $\mathbf{L}_j^{-1}$ usando una función genérica (`np.linalg.inv`) y la almacena. En `predict`, realiza la multiplicación matricial $\mathbf{y}_j = \mathbf{L}_j^{-1} \boldsymbol{\delta}_j$.

*   **`QDA_Chol2`:** Adopta la práctica numéricamente más recomendada: evitar la formación explícita de la inversa. En lugar de calcular $\mathbf{y}_j = \mathbf{L}_j^{-1} \boldsymbol{\delta}_j$, resuelve el sistema de ecuaciones lineales triangulares $\mathbf{L}_j \mathbf{y}_j = \boldsymbol{\delta}_j$. Dado que $\mathbf{L}_j$ es triangular inferior, este sistema se resuelve de manera muy eficiente mediante **sustitución hacia adelante (forward substitution)**, implementada en `scipy.linalg.solve_triangular`.

*   **`QDA_Chol3`:** Es un híbrido. Al igual que `QDA_Chol1`, calcula y almacena explícitamente la inversa $\mathbf{L}_j^{-1}$. Sin embargo, en lugar de usar una rutina de inversión genérica, utiliza `dtrtri` de LAPACK, una función altamente optimizada y especializada para invertir matrices triangulares. Se espera que sea más rápida y precisa que la inversión de propósito general para este caso específico.