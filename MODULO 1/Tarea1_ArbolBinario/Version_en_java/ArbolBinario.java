package arbolBinario;

public class ArbolBinario {
	private Nodo raiz;
	
	public ArbolBinario() { //metodo para inicializar
        this.raiz = null;
    }
	
	public boolean buscar(String busco) {
		boolean encontrado = buscarRecursivo(raiz, busco);
		return encontrado;
	}
	
	private boolean buscarRecursivo(Nodo raiz, String busco) {
		if(raiz==null) 
			return false; //caso base no hay para donde buscar
		
		String valor = raiz.getValor();
		
		if(busco.equals(valor)) 
			return true;
		
		if(busco.compareTo(valor) < 0) 
			return buscarRecursivo(raiz.getIzquierdo(), busco);
		else
			return buscarRecursivo(raiz.getDerecho(), busco);
			
	}
	
	public void insertar(String valor) {
		//En este metodo determinaremos la nueva raiz
		raiz = insertarRecursivo(raiz, valor);
	}
	
	private Nodo insertarRecursivo(Nodo raiz, String valor) {
		//Caso base arbol vacio
		if (raiz == null) {
            raiz = new Nodo(valor);
            return raiz;
        }
		
		//Insertar para arbol balanceado ok
		
		if (valor.compareTo(raiz.getValor()) < 0) {
			//En el caso de que es menor me voy a la izquierda e intento insertar ahí
			//El nodo solo se pondrá hasta que esté ejecutandose el caso base
			Nodo izquierdoActual = raiz.getIzquierdo();
            raiz.setIzquierdo(insertarRecursivo(izquierdoActual, valor));
            
        } else if (valor.compareTo(raiz.getValor()) > 0) {
        	Nodo derechoActual = raiz.getDerecho();
            raiz.setDerecho(insertarRecursivo(derechoActual, valor));
        }
		
		return raiz;
	}
	
	public void imprimir() {
		System.out.println("Imprimiendo arbol");
		System.out.println("-----------------");
		imprimirRecursivo(raiz);
	}
	
	private void imprimirRecursivo(Nodo raiz) {
		//Inorden??
		if(raiz!=null) {
			imprimirRecursivo(raiz.getIzquierdo());
			System.out.println(raiz.getValor());
			imprimirRecursivo(raiz.getDerecho());
		}
	}	
}
