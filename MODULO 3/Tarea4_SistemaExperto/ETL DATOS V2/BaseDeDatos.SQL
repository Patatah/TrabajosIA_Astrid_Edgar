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

--INDICES
CREATE INDEX IX_Ingredientes_nombre ON Ingredientes(nombre);
CREATE INDEX IX_RecetaIngrediente_Ingrediente_Receta ON RecetaIngrediente(idIngrediente, idReceta);
CREATE INDEX IX_RecetaIngrediente_Receta_Ingrediente ON RecetaIngrediente(idReceta, idIngrediente);
CREATE INDEX IX_Ingredientes_ocurrencias ON Ingredientes(ocurrencias DESC) INCLUDE (nombre);
GO

--consultas
select * FROM recetas
select * from ingredientes
select count(*) as ingredientesUsados from RecetaIngrediente
select * from RecetaIngrediente


--Recolectar ingredientes de receta
SELECT I.nombre AS Ingrediente, I.idIngrediente as ID
FROM Recetas R
JOIN RecetaIngrediente RI ON R.idReceta = RI.idReceta
JOIN Ingredientes I ON RI.idIngrediente = I.idIngrediente
--WHERE R.idReceta = 67664;

--Recolectar recetas de un ingrediente
SELECT R.idReceta, R.nombre AS NReceta, STRING_AGG(I.nombre, ', ') AS Ingredientes
FROM Recetas R
JOIN RecetaIngrediente RI ON R.idReceta = RI.idReceta
JOIN Ingredientes I ON RI.idIngrediente = I.idIngrediente
WHERE EXISTS (
    SELECT 1
    FROM RecetaIngrediente RI
    JOIN Ingredientes I ON RI.idIngrediente = I.idIngrediente
    WHERE RI.idReceta = R.idReceta AND I.nombre = 'milk'
)
AND EXISTS (
    SELECT 1
    FROM RecetaIngrediente RI
    JOIN Ingredientes I ON RI.idIngrediente = I.idIngrediente
    WHERE RI.idReceta = R.idReceta AND I.nombre = 'flour'
)
GROUP BY R.idReceta, R.nombre;

--Recolectar que ingredientes te faltan para hacer algo
SELECT DISTINCT I.nombre, I.ocurrencias AS Ocurrencias
FROM Ingredientes I
JOIN RecetaIngrediente RI ON I.idIngrediente = RI.idIngrediente
WHERE RI.idReceta IN (
    -- Recetas que contienen ambos: leche Y harina
    SELECT R.idReceta
    FROM Recetas R
    WHERE EXISTS (
        SELECT 1
        FROM RecetaIngrediente RI
        JOIN Ingredientes I ON RI.idIngrediente = I.idIngrediente
        WHERE RI.idReceta = R.idReceta AND I.nombre = 'milk'
    )
    AND EXISTS (
        SELECT 1
        FROM RecetaIngrediente RI
        JOIN Ingredientes I ON RI.idIngrediente = I.idIngrediente
        WHERE RI.idReceta = R.idReceta AND I.nombre = 'flour'
    )
)
ORDER BY 
    I.ocurrencias DESC, 
    I.nombre ASC;       

-- Todas las recetas que puedes hacer con esos ingredientes que tienes
SELECT *
FROM Recetas r
WHERE (
    SELECT COUNT(DISTINCT i.nombre)
    FROM RecetaIngrediente ri
    JOIN Ingredientes i ON ri.idIngrediente = i.idIngrediente
    WHERE ri.idReceta = r.idReceta
    AND i.nombre IN ('cornstarch', 'bacon', 'ground beef')
) = 3;  