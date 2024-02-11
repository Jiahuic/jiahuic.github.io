---
layout: page
permalink: /teaching/math-499v599v/lec1/
title: Introduction to Topological Data Analysis
---

# Topological Data Analysis (TDA)
Topological data analysis (TDA) is a collection of methods for analyzing datasets that have an underlying topological structure. The goal of TDA is to describe the shape of data, and to find the best possible summary (invarient) of that shape in a way that is robust to noise and other "nuisance" parameters. TDA is a relatively new field, and is an active area of research in applied mathematics and statistics.

## Topological Spaces
A topological space is a set $X$ together with a collection of subsets of $X$ called open sets. The open sets must satisfy the following axioms:
1. The empty set $\emptyset$ and $X$ are open.
2. The union of any collection of open sets is open.
3. The intersection of any finite collection of open sets is open.

A topological space is a generalization of the notion of an open subset of $\mathbb{R}^n$. For example, the open interval $(0,1)$ is an open subset of $\mathbb{R}$, and the open ball $B(0,1)$ is an open subset of $\mathbb{R}^2$. The collection of all open subsets of $\mathbb{R}^n$ is called the standard topology on $\mathbb{R}^n$.

## Simplicial Complexes
A simplicial complex is a topological space that is constructed from a collection of simplices. A simplex is a generalization of a triangle or tetrahedron to arbitrary dimensions. A $0$-simplex is a point, a $1$-simplex is a line segment, a $2$-simplex is a triangle, a $3$-simplex is a tetrahedron, and so on. A $k$-simplex is the convex hull of $k+1$ affinely independent points in $\mathbb{R}^n$. For example, a $2$-simplex is the convex hull of $3$ affinely independent points in $\mathbb{R}^2$, and is therefore a triangle.

## Persistent Homology
Persistent homology is a method for computing topological invariants of a dataset. The basic idea is to construct a simplicial complex from the data, and then compute the homology groups of the simplicial complex. The homology groups are a collection of invariants that describe the shape of the simplicial complex. The homology groups are a sequence of vector spaces, and the dimension of each vector space is called the Betti number. The Betti numbers are the topological invariants that we are interested in.

The homology groups are computed by constructing a chain complex of vector spaces. The chain complex is a sequence of vector spaces and linear maps between them. The homology groups are the vector spaces that are the kernels of the linear maps. The chain complex is constructed from the simplicial complex by assigning a vector space to each simplex, and a linear map to each pair of simplices. The vector space assigned to a simplex is called the chain group of the simplex, and the linear map assigned to a pair of simplices is called the boundary map of the pair of simplices.

The chain complex is constructed by assigning a vector space to each simplex, and a linear map to each pair of simplices. The vector space assigned to a simplex is called the chain group of the simplex, and the linear map assigned to a pair of simplices is called the boundary map of the pair of simplices. The chain complex is constructed by assigning a vector space to each simplex, and a linear map to each pair of simplices. The vector space assigned to a simplex is called the chain group of the simplex, and the linear map assigned to a pair of simplices is called the boundary map of the pair of simplices.

## Persistent Homology of a Point Cloud
The persistent homology of a point cloud is a method for computing the homology groups of a simplicial complex. The simplicial complex is constructed from the point cloud by constructing a simplex for each point in the point cloud, and then constructing a simplex for each pair of points in the point cloud. The simplicial complex is constructed by constructing a simplex for each point in the point cloud, and then constructing a simplex for each pair of points in the point cloud. The simplicial complex is constructed by constructing a simplex for each point in the point cloud, and then constructing a simplex for each pair of points in the point cloud. The simplicial complex is constructed by constructing a simplex for each point in the point cloud, and then constructing a simplex for each pair of points in the point cloud. The simplicial complex is constructed by constructing a simplex for each point in the point cloud, and then constructing a simplex for each pair of points in the point cloud. The simplicial complex is constructed by constructing a simplex for each point in the point cloud, and then constructing a simplex for each pair of points in the point cloud.

