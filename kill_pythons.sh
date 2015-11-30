#!/bin/sh

#kill all python processes

kill -9 `ps  -u ruiliu |grep python |awk '{print $1}'`
