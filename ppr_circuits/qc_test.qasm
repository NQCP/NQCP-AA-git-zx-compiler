OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
h q[0];
t q[0];
cx q[1],q[2];
t q[1];
x q[2];