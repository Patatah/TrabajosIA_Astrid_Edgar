CREATE DATABASE CooklyDB;
GO

USE CooklyDB;
GO

-- Tabla de recetas
CREATE TABLE Recetas (
    idReceta INT PRIMARY KEY,
    nombre VARCHAR(255),
    minutos INT,
    fecha DATE,
    n_pasos INT,
    pasos TEXT,
    n_ingredientes INT,
    descripcion TEXT
);

CREATE TABLE Ingredientes (
    idIngrediente INT PRIMARY KEY,
    nombre VARCHAR(255),
    ocurrencias INT
);

CREATE TABLE RecetaIngrediente(
    idReceta INT,
    idIngrediente INT,
    FOREIGN KEY (idReceta) REFERENCES Recetas(idReceta),
    FOREIGN KEY (idIngrediente) REFERENCES Ingredientes(idIngrediente),
    PRIMARY KEY (idReceta, idIngrediente)
)
GO

--consultas
select nombre FROM recetas
select nombre from ingredientes
select count(*) as ingredientesUsados from RecetaIngrediente

--Recolectar ingredientes de una receta
SELECT I.nombre AS Ingrediente
FROM Recetas R
JOIN RecetaIngrediente RI ON R.idReceta = RI.idReceta
JOIN Ingredientes I ON RI.idIngrediente = I.idIngrediente
WHERE R.idReceta = 137739;