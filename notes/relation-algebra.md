## Cartesian Product

The result of a Cartesian product has an arity that is the sum of the arities of the operands.

The result of a Cartesian product has the number of tuples that is the product of the tuple numbers of the operands.

```
In a Cartesian product T := R *  S

NumberOfColumns(T) = NumberOfColumns(R) + NumberOfRows(S)

NumberOfRows(T) = NumberOfRows(R) * NumberOfRows(S)
```

## Division

A division is like the reverse of a product.  For a division R3 / R2 that produces R1, the number of attributes is:

```
ncol(R1) = ncol(R3) - ncol(R2).  
```
The number of observations is irrelevant.

## Union

A union requires that both operands have the same arity (and schema).
Therefore R3 has the same arity as σ{ ϕ }(R1 X R2).

## Selection
The selection, σ{ ϕ }, does not change the arity. Therefore R3 has the same arity as R1 * R2. The arity of R1 X R2 is the arity of R1 plus the arity of R2.

## Projection

Unary means "one". A projection chooses columns from only one relation.

The projection π{ C1 } results in a relation that has only the column
“C1”.

## Intersection

An intersection finds commonalities between "two" tables
An Intersection is binary but it does not combine tuples.

An intersection with R3 is only possible if the two operands
have the same schema. Therefore R3 has only one column.

## Join


