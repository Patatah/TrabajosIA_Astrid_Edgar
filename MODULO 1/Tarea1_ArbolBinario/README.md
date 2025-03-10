**Explicación del proyecto**

En este proyecto creamos un **árbol binario de búsqueda**.  
Un árbol binario es una estructura de datos. En este caso, almacenamos cadenas en cada nodo. Además, almacenamos dos objetos de tipo `Nodo` para hacer referencia a las conexiones que tiene. Las iniciamos ambas vacías y, al ir insertando, se van agregando.  

Por la parte de la clase `ArbolBinario`, lo único que contiene es la raíz del árbol. Esto es porque, a partir de la raíz, podemos acceder a cualquier punto del árbol con métodos recursivos de manera sencilla.  

Para inicializar el árbol, solo creamos una raíz vacía, y eso es todo.  

Para insertar, lo hacemos de manera recursiva. Si el método encuentra un lugar vacío donde colocar el nodo, lo pone ahí (este es el caso base). Si no, compara el dato que intentas insertar para determinar si debe repetir el método hacia la izquierda o hacia la derecha hasta que el nodo encuentre un lugar.  

Este tipo de árboles es muy eficiente al buscar algo en específico. El método devuelve `true` si encuentra lo que buscas y funciona comparando el dato con el nodo actual, procediendo de cuatro formas:  
- Si estoy en un nodo vacío, no encontré lo que buscaba.  
- Si estoy en un nodo que es igual, encontré lo que buscaba.  
- Si estoy en un nodo mayor que lo que quiero, voy hacia la izquierda.  
- Si estoy en un nodo menor que lo que quiero, voy hacia la derecha.  

En cuanto a la impresión, es tan simple como imprimir el nodo izquierdo, luego el actual y luego el derecho para hacer un recorrido inorden.  

Todo esto está probado en el `main` con un árbol sencillo.
