package arbolBinario;

public class Main {
	
	public static void main(String args[]) {
		//Inicializar
		ArbolBinario arbolito = new ArbolBinario();
		
		//Insertar
		arbolito.insertar("Edgar");
		arbolito.insertar("Astrid");
		arbolito.insertar("Octavio");
		arbolito.insertar("Diego");
		arbolito.insertar("Brayan");
		arbolito.insertar("Chiquete");
		arbolito.insertar("Omar");
		arbolito.insertar("Ricardo");
		arbolito.insertar("Ximena");
		arbolito.insertar("Joselyn");
		arbolito.insertar("Andres");
		
		//Buscar
		String busqueda = "Edgar";
		if(arbolito.buscar(busqueda)) 
			System.out.println("Encontrado "+busqueda);
		else
			System.out.println("No se encontró "+busqueda);
		
		System.out.println();
		//Imprimir
		arbolito.imprimir();
	}
}
