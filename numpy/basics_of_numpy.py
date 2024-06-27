import numpy as np

"""
INTRODUCTION

Numpy es una libería para manejar arreglos multidimensionales en python.

Lo que hace a numpy muy potente es que implementa algoritmos en c y usa a python cómo un 
pegamento o interfaz para trabajar con los arreglos. Pero lo mas importante es que dichas 
implementaciones usan bloques contiguos de memoria lo que ayuda mucho a que la CPU pueda 
hacer un cache mucho mas eficientemente, a diferencia de las listas de python que se 
implementan como arreglos de punteros a locaciones de memorias aleatorias, lo que dificulta
el proceso de caching para la CPU lo que conlleva a un acceso a memoria mucho mas costoso
(Muchos mas accesos por que la información no esta en los cache del cpu)

Sin embargo, el poco acceso a memoria y por ende la ganancia en eficiencia computacional 
tienen un costo: los arreglos de numpy tienen el mismo tamaño y son homogeneos, lo que 
significa que todos los elementos tienen el mismos tipo. Esto le da una ventaja a numpy
de poder usar loops eficientes de C y evitar costos chequeos de tipo de datos y otras 
sobrecargas del api de listas de python.

Una cosa a tomar en cuenta es que agregar y quirar elementos al final de una lista
de python es muy eficiente, alterar el tamaño de una array de numpy es muy costoso por
que requiere crear un nuevo arreglo y trasladar el contenido del arreglo anterior al nuevo 
que queremos expandir o reducir.

Además de ser mas eficiente para computaciones numéricas que el código de python nativo, 
numpy también es mas elegante y mas legible debido a sus operaciones de vectores que ofrece. 
"""



"""
N-dimensional Arrays

Numpy está construido con ndarrays que son estructuras de datos de alto rendimiento.
Podemos pensar en un arreglo de una dimensión (1-dimensional array) como una estructura
de datos que representa un vector de elementos, puede verse cómo una lista de python
dónde todos los elementos son del mismso tipo. De igualmanera se puede pensar en 
un arreglo de dos dimensiones (2-dimensional array) como una estructura de datos que 
representa a una matriz o bien como una lista de listas en python.

"""

"""
Para crear un arreglo de numpy podemos usar la function array. Ahora vamos a crear un 
arreglo de 2 dimensiones a traves de una lista de lista de python 
"""

lst = [[1,2,3],
       [4,5,6]]
array = np.array(lst)

#Por default python infiere el tipo de dato, en este caso int64 para una máquina de 64 bits
#Se puede confirmar con el atributo dtype
array.dtype #-> 'int64'

#Podemos espesificar el tipo de dato al crear el arreglo
#Existe una lista de tipos soportados en la documentación oficial de numpy
array = np.array(lst, dtype=np.float32)
array.dtype   # -> 'float32'

#Se puede cambiar el tipo de dato de un arreglo existente con astype
#Como se menciono anteriormente, la eficiencia de numpy tiene un costo y para hacer esto se tiene que crear un nuevo arreglo
new_array = array.astype(np.float64)
print(new_array.dtype)


#A continuación se presentan algunos atributos los arrays de numpy

array.itemsize #Regresa el tamaña de un elemento en bytes (4 en este caso. lo que nos da los 32 bits del tipo de dato del arreglo )
array.size #Regresa el número total de elementos del arreglo (6 en esta caso)
array.ndim #Regrea el número de dimensiones de un arreglo (2 en este casoi)
array.shape #Regresa el número de elementos de cada dimension (en el contexto de numpy nos vamos a referir a ellos como axes)
#El atributo shape es siempre una tupla, para arreglos de una sola dimensión regresa una tupla con el número de elementos, para
#arreglos de mas dimensiones regresa una tupla de 2 elementos: el primero es el numero de rows y el segundo el numero de columnas. 
#No se debe de confundir el número de columnas y/o filas con el número de dimensiones ya que se puede tener una arreglo de 2x3(2 filas 3 columnas) y uno de 5x4 (5 filas, 4 columas) y ambos arreglos siguen siendo de 2 dimensiones


#También se pueden crear arreglos de dimensión cero, lo que en realida no sería un arreglo si no un valor escalar
scalar = np.array(5)
print(scalar.ndim) #-> 0
print(scalar.shape) # ()  una tupla vacía



"""
    RUTINAS DE CONTRUCCIÓN DE ARREGLOS

    Como vimos podemos usar la funcion array de numpy para crear numpy arrays, la cual puede recibir, listas, tuplas, listas anidadas, rangos,etc
    Si bien la función array es un buen punto de partida no es la única manera de crear arreglos de numpy, existen otras funciones que nos dan la vent    aja de crear arreglos mediante expresiones generadoras (generators o generator expressions), lo cual np.array no permite.
    A continuación vamos a ver las funciones mas comunes para crear los arreglos

"""

"""
    np.fromiter -> Nos permite crear el arreglo en base a generadores e iteradores
"""
def generator():
    for i in range(10):
        if i % 2:
            yield i

gen = generator()
np.fromiter(gen, dtype='int')
#array([1, 3, 5, 7, 9])

#Podríamos reemplazar el generador usando python 'comprehensions'
gen = (i for i in range(10) if i % 2)
np.fromiter(gen, dtype='int')
#array([1, 3, 5, 7, 9])


"""
    np.ones y np.zeros 
    Funciones para crear matrices de puros unos y puros ceros respectivamente. Hay que especificar el número de filas y columnas
    Creating arrays of ones or zeros can also be useful as placeholder arrays,
in cases where we do not want to use the initial values for computations
but want to fill it with other values right away. If we do not need the initial
values (for instance, ’0.’ or ’1.’), there is also numpy.empty, which
follows the same syntax as numpy.ones and np.zeros. However, instead
of filling the array with a particular value, the empty function creates the
array with non-sensical values from memory. We can think of zeros as a
function that creates the array via empty and then sets all its values to 0.
– in practice, a difference in speed is not noticeable, though.

"""
np.ones((3,4)) #matriz de puros unos de 3 filas 4 columnas 
np.zeros((3,3)) #Matriz de puros ceros de 3 filas 3 columnas

"""
    np.eye y np.diag
    Funciones para crear marrices identidad (una matriz diagonal con unos) y matrices diagonal
"""

np.eye(3) #Crea una matriz identidad de 3x3
"""
    [1,0,0]
    [0,1,0]
    [0,0,1]
"""
np.diag([3,3,3]) # ->
"""
    [3,0,0]
    [0,3,0]
    [0,0,3]
"""


"""
    np.arange y np.linspace
    Funciones para crear secuencias de datos a partir de un rango específico. arange sigue la misma sintaxis de range quye python (valor inicial, valor final, incremento). linspace crea una secuencia a partir de un valor de incremento y un rango especificado, es decir, el rango lo divide en elementos iguales con el valor de incremento

"""
np.arange(4,10) # array(4,5,6,7,8,9)
np.arange(5) #array(0,1,2,3,4)
np.arange(1,11,2) #array(1,3,5,7,9)
np.linspace(0,1,num=5) #Divide el rango de tal manera que quede en 5 elemntos y los incrementos sean uniformes -> array(0,0.25,0.5,0.75,1)


"""
ARRAY INDEXING

En esta sección vamos a tratar las bases para accesar a los elementos de los arreglos. Podremos Observar que es muy similar a python

"""

a = np.array([1,2,3])
a[0] # -> 1

"""
slicing operations
"""

print(a[:2]) #-> obtiene los dos primeros elementos [1,2], equivalente a [0:2]
print(a[0:2])

"""
Si estamos trabajando con arreglos multidimensionales, separeremos nuestras operaciones de indexado y slicing con comas para acceder a los elementos
de los arreglos 
"""

a = np.array(lst)
a[0, 0] # Primer elemento (uper, left) -> 1
a[-1,-1] #Último elemento (lower, right) -> 6
a[0, 1] #First row, second column -> 2
a[0] #La primer fila completa -> [1,2,3]
a[:, 0] #La primer columna completa -> [1,4]
a[:, :2] #Las primeras dos columnas completas -> [[1,2]
                                            #     [4,5]]



"""
    ARRAY MATH AND UNIVERSAL FUNCTIONS

    Numpy proporciona algo llamado ufuncts que son vectorized wrappers (código que ejecuta operaciones con vectores) para realizar operaciones
    sobre los elementos de los arreglos. Existen mas de 60 ufuncts disponibles implementadas en C y son muy rapidas y eficientes comparadas con 
    el código que podríamos escribir en python.
    Por ejemplo si quisieramos sumar 1 a cada elemento de una matriz podríamos usar un código de python cómo el siguiente:
"""

for row_idx, row_val in enumerate(lst):
    for col_idx, col_val in enumerate(lst):
        lst[row_idx][col_idx] += 1

"""
    Podemos observar que este código es muy verboso, puíeramos reescribirlo de una manera mas elegante con 
    lists comprehensions de la siguiente manera
"""

[[cell + 1 for cell in row] for row in lst]

"""
Podríamos realizar lo mismo utilizando las ufuncs de numpy de una manera mucho mas sencilla y eficiente
"""

a = np.array(lst)
new_ary = np.add(a, 1)
"""
    new_ary -> [[2,3,4]
                 [5,6,7]]

    las ufuncts para operaciopnes básicas son: add, substract, divide, multiply y exp. Sin embargo numpy implemeta sobrecarga de operadores con lo que podemos usar las operaciones matemáticas directamente. Por ejemplo
"""
a**2
new_ary = a + 1


"""
    new_ary -> [[2,3,4]
                 [5,6,7]]

    En los ejemplos de código anteriores (add y +) usuamos binary ufuncts que son funciones que realizan cálculos entre 2 argumentos de entrada.
    Numpy también implementa unary ufunct que relizan cálculos en eñ arreglo, dichas funciones son: log, log10, y sqrt
"""
np.sqrt(a)

"""
    A menudo vamos a querer calcular la suma o el producto de los elementos de una matriz a lo largo de un eje dado. Para esto podemos usar la función reduce. Por default reduce aplica la operación sobre el primer eje (axis 0). En el caso de una matriz bidimensional podemos pensar en el primer eje
    como las filas, por lo tanto al sumar elementos a lo largo de las filas se obtienen las sumas de las columnas. Se explica con el siguiente
    fragmento de código.
"""

lst = [[1,2,3],
       [4,5,6]]

a = np.array(lst)

np.add.reduce(a) # -> [5,7,9]
np.add.reduce(a, axis = 1) # -> [6,15]


#Numpy provee de atajos para opereaciones de producto y suma que serían así:
np.sum(a, axis = 0) # -> [5,7,9]
np.sum(a,axis = 1) # -> [6,15]

#Si no se especifica el axis numpy sumara los elementos de todo el arreglo
np.sum(a) #-> 21
np.product(a) # -> 720

"""
     otras ufuncs son:

     mean -> Calcula el promedio artimético
     std -> Calcula la desviación estandar
     var -> Calcula la varianza
     sort -> ordena un arreglo
     argsort -> Regresa los indices del arreglo ordenado
     min
     max
     argmin
     argmax
     array_equal -> Verifica si dos arreglos tienen el mismo shape (rows y columns) y los mismos elementos
"""

"""
BROADCASTING

Es la manera en que numpy nos permite reazar operaciones de vectores entre arreglos de diferentes dimensiones,
esto lo hace creando arrays implicitos para rellenar los datos faltantes. Anteriormente aprendimos como hacer 
operaciones con las ufuncts, que es un ejemplo del broadcasting.
"""

a1 = np.array([1,2,3])
a2 = np.array([[1,2,3],
               [4,5,6]])

np.add(a1, a2)
"""
Es cómo si sumaramos [[1,2,3],   +   [[1,2,3],
                      [1,2,3]]        [4,5,6]]

                    =  [[2,4,6],
                        [5,7,9]]

Debemos tomar en cuenta que el número de elementos del eje del arreglo que se va a crear implicitamente 
tiene que coincidir con el arreglo de mayor dimensión. Ejemplo
"""

a_2d = np.array([[1,2,3],
               [4,5,6]])
a_1d = np.array([1,2])

#Si sumamos estos arreglos vamos a obtener un error por que coinciden el número de elementos del eje 0

try:
    a_2d + a_1d
except Exception as e:
    print(e)
    #operands could not be broadcast together with shapes (2,3) (2,)



"""
ADVANCED INDEXING: MEMORY VIEWS AND COPIES

Cuando hacemos indexado básico con números enteros creamos views. Las views acceden al mismos arreglo en
memoria pero lo ven (de aquí el nombre) de manera diferente(diferentes shapes o elementos por ejemplo).
Trabajar con vistas es muy deseable ya que evitan la creación de copias de los arreglos en memoria.
Veamos algunos ejemplos
"""

a = np.array([[1,2,3], 
              [4,5,6]])

first_row = a[0] #first_row es una vista del arreglo a. Accede a las mismas ubicaciones de memoria pero es como si fuera otro arreglo diferente
first_row + 100  
"""
    a =  [[101,102,103],
          [4,5,6]]

    Este mismo concepto aplica al indexado mediante slicing, por ejemplo a[:, 2] (Se crea una vista de la tercera columa)

    En ciertos escenarios vamos a querer crear copias de los arreglos para que las modificaciones
    que hagamos no afecten al arreglo original, esto lo podemos hacer con el método copy. Ejemplo:
"""
a = np.array([[1,2,3], 
              [4,5,6]])
second_row = a[1].copy()
second_row += 100
"""
    a =  [[1,2,3],
          [4,5,6]]

    Una manera de saber si dos arreglos comprarten memoria es usar la función may_share_memory,
    se debe de tener cuidado ya que en raros casos puede dar resultados erroneos.
    Veamos unos ejemplos de may_share_memory
"""

a1 = np.array([[1,2,3], [4,5,6]])
a2 = a1[:, 2] #Tercera columna
np.may_share_memory(a1,a2) #True
a2 = a1[:, 2].copy() #Tercera columna, pero con un nuevo arreglo en memoria
np.may_share_memory(a1,a2) #False

"""
    Además de la indexado baśico con enteros y slicing numpy ofrece rutinas avanzadas de indexado 
    llamadas fancy indexing. Con fancy indexing podemos usar tuplas o listas de enteros no contiguos.
    El indexado fancy siempre regresa copias de los arreglos, nunca vistas
"""
a = np.array([[1, 2, 3],
            [4, 5, 6]])

new_array = a[:, [0,2]] #Primera y última columna

"""
    También podemos usar mascaras boolenas (Boolean Mask) para el indexado, esto sería arreglos 
    de valores True y False. Ejemplo:
"""
a = np.array([[1, 2, 3],
            [4, 5, 6]])
mayores_a_3 = a > 3
"""
mayores_a_3 = [[False, False, False],
               [True, True, True]]

    Usando estas mascaras podemos seleccionar los valores que deseamos como en el siguiente ejemplo
"""
a[mayores_a_3] # -> [4,5,6]
"""
    Podemos crear criterios de selección mas complejos usando los operadores and y or, & y | 
    respectivamente. El siguiente ejemplo selecciona los elementos mayores que 3 y divisibles 
    por 2:
"""
a[(a > 3) & ( a % 2 == 0)] # [4,6]


"""
COMPARISON OPERATORS AND MASK

Anteriormente tuvimos una breve introducción al concepto de boolean masks, que son arreglos de tipo boolean
que tienen el mismo shape que un arreglo de etiqueta. Usando operadores de comparación (<, >, >=, <=) 
podemos crear mascaras booleanas, una ves creada las mascaras podemos seleccionar ciertos elementos de
nuestro arreglo original (target array).
Con las mascaras podemos contar los elementos que coiciden con la condición, podemos obtener los indices de 
los elementos que coinciden como se muestra a continuación 
"""
a = np.array([1,2,3,4])
mask = a > 2
mask.sum() # 2
mask.nonzero() # [2,3] los elementod del arreglo regresado son los indices que coinciden con las condición de la mascara

"""
Alternativamente podemos seleccionar los indices que cumplen con una condición usando el método where
"""
np.where(a > 2) # [2,3]
 #Weher puede recibir 3 parámetros el primero es la condición, el segundo un valor para los verdaderos y el tercero un valor para los falsos
np.where(a > 2, 1, -1) # [-1, -1, 1, 1]
#Una manera de hacerlo con mascaras sería
a[mask] = 1
a[~mask] = -1
"""
El operador ~ presentado anteriormente es uno de los operadores lógicos de numpy:

    and: & ó np.bitwise_and
    or: | ó np.bitwise_or
    xor: ^ ó np.bitwise_xor
    not: ~ ó np.bitwise_not

    Los peradores lógicos nos permiten encadenar comparaciones para crear mascaras booleanas mas complejas
"""

print('Fin de archivo')