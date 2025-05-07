CREATE DATABASE IA;
USE IA;

-- Tabla de recetas
CREATE TABLE Recetas (
    id INT PRIMARY KEY,
    nombre VARCHAR(255),
    minutos INT,
    fecha DATE,
    n_pasos INT,
    pasos TEXT,
    n_ingredientes INT,
    ingredientes TEXT,
    descripcion TEXT,
);