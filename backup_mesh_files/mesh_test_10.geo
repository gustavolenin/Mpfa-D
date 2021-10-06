// Gmsh project created on Wed Feb 14 11:47:48 2018
Point(1) = {0, 0, 0, 1.0};
Point(2) = {1, 0, 0, 1.0};
Point(3) = {1, 1, 0, 1.0};
Point(4) = {0, 1, 0, 1.0};

Point(5) = {0, 0, 1, 1.0};
Point(6) = {1, 0, 1, 1.0};
Point(7) = {1, 1, 1, 1.0};
Point(8) = {0, 1, 1, 1.0};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

Line(5) = {5, 6};
Line(6) = {6, 7};
Line(7) = {7, 8};
Line(8) = {8, 5};

Line(9) = {1, 5};
Line(10) = {2, 6};
Line(11) = {3, 7};
Line(12) = {4, 8};

Line Loop(13) = {1, 2, 3, 4};
Plane Surface(14) = {13};

Line Loop(15) = {5, 6, 7, 8};
Plane Surface(16) = {15};

Line Loop(17) = {9, 5, -10, -1};
Plane Surface(18) = {17};

Line Loop(19) = {10, 6, -11, -2};
Plane Surface(20) = {19};

Line Loop(21) = {11, 7, -12, -3};
Plane Surface(22) = {21};

Line Loop(23) = {12, 8, -9, -4};
Plane Surface(24) = {23};

Surface Loop(25) = {22, 20, 18, 24, 16, 14};
Volume(26) = {25};

mesh_size = 1.5;

Physical Point(102) = {1, 4, 8, 5};
Physical Point(202) = {2, 3, 7, 6};

Physical Surface(201) = {18, 14, 22, 16};
Physical Surface(102) = {24};
Physical Surface(202) = {20};

Physical Volume(1) = {26};
//+
Physical Point(102) -= {8};
//+
Physical Point(102) -= {4};
//+
Physical Point(102) -= {5};
//+
Physical Point(102) -= {1};
//+
Physical Point(202) -= {3, 7, 6, 2};
//+
Physical Surface(102) -= {16, 20, 22, 14, 24, 18};
//+
Physical Surface(201) -= {20, 22, 14, 16, 18};
//+
Physical Surface(202) -= {20};
//+
Physical Surface("Neumann", 201) = {16, 22, 24, 20, 14, 18};
//+
Physical Point(101) = {1};
//+
Physical Point(102) = {7};
//+
Physical Surface(" Neumann", 201) -= {24};
//+
Physical Surface(" Neumann", 201) -= {20};
//+
Physical Surface("Dirichlet", 101) = {24};
//+
Physical Surface("Dirichlet", 102) = {20};
//+
Physical Surface("Dirichlet", 102) = {20};
//+
Physical Surface("Dirichlet", 102) = {20};
//+
Physical Surface("Dirichlet", 101) += {20};
//+
Physical Surface("Dirichlet", 102) = {20};
//+
Physical Surface("Dirichlet", 102) = {20};
//+
Physical Surface(" Dirichlet", 101) -= {20};
//+
Physical Surface("Dirichlet", 102) = {20};
//+
Physical Surface(102) = {20};
