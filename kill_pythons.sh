#!/bin/sh

#kill all python processes

kill -9 `ps  -u rui |grep python |awk '{print $1}'`
