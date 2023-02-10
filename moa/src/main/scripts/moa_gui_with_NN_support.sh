#!/bin/bash
# ----------------------------------------------------------------------------
#  Copyright 2001-2006 The Apache Software Foundation.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# ----------------------------------------------------------------------------

#   Copyright (c) 2001-2002 The Apache Software Foundation.  All rights
#   reserved.

#   Copyright (C) 2011-2019 University of Waikato, Hamilton, NZ

print_usage()
{
  echo "Usage: $0 <local_maven_repo> <conda env> <djl_cache_dir>"
  echo "e.g:   $0 ~/Desktop/m2_cache/ /Users/ng98/Desktop/condaJava ~/Desktop/djl.ai/"
}

BASEDIR=`dirname $0`/..
BASEDIR=`(cd "$BASEDIR"; pwd)`
REPO=$BASEDIR/../../target/classes

if [ $# -lt 2 ]; then
    print_usage
    exit 1
fi

if [ -d "$1" ]; then
  MAVEN_REPO="$1"
  export MAVEN_OPTS="-Dmaven.repo.local=$1"
else
  echo "MAVEN_OPTS=-Dmaven.repo.local can not be set. Directory $1 is not available."
  print_usage
  exit 1
fi


eval "$(conda shell.bash hook)"
conda init bash
echo "Running: conda activate $2"
conda activate "$2"
if [ $? -ne 0 ]; then
  echo "conda activate $2 Failed."
  echo "available conda environments:"
  conda env list
  print_usage
  exit 1
fi

if [ -d "$3" ]; then
  export DJL_CACHE_DIR=$3
else
  echo "DJL_CACHE_DIR can not be set. Directory $3 is not available. Creating it"
  mkdir -p $3
fi


#JAR_PATHS="$(for j in $(find $MAVEN_REPO -name '*.jar'| grep -v 'kafka'| grep -v 'moa-2021.07.0.jar');do printf '%s:' $j; done)"
#JAR_PATHS="$(find $MAVEN_REPO -name '*moa-*SNAPSHOT.jar'| grep -v 'kafka')"
#CLASSPATH="$JAR_PATHS$REPO/"
#JAR_PATHS="$(for j in $(find ${MAVEN_REPO}com/github/waikato/ -name '*.jar');do printf '%s:' $j; done)$JAR_PATHS:"
#JAR_PATHS="$(for j in $(find ${MAVEN_REPO}com/jidesoft/ -name '*.jar');do printf '%s:' $j; done)$JAR_PATHS"

#JAR_PATHS="$(find $MAVEN_REPO -name '*moa-*SNAPSHOT.jar'| grep -v 'kafka')"
#JAR_PATHS="$(for j in $(find ${MAVEN_REPO} -name '*.jar'|grep -v 'SNAPSHOT.jar' |grep -v 'kafka');do printf '%s:' $j; done)$JAR_PATHS"
#JAR_PATHS="$JAR_PATHS:$(for j in $(find ${MAVEN_REPO} -name '*.jar');do printf '%s:' $j; done)"
#JAR_PATHS="$JAR_PATHS:$(for j in $(find ${MAVEN_REPO}com/fifesoft/ -name '*.jar');do printf '%s:' $j; done)"
#CLASSPATH="$JAR_PATHS$REPO/"

JAR_PATHS="$(find $MAVEN_REPO -name '*moa-*SNAPSHOT.jar'| grep -v 'kafka')"
JAR_PATHS="$(for j in $(find ${MAVEN_REPO} -name '*.jar'| grep -v 'moa');do printf '%s:' $j; done)$JAR_PATHS"
CLASSPATH="$JAR_PATHS"
echo"$CLASSPATH"
JAVA_AGENT_PATH="$(find $MAVEN_REPO -name 'sizeofag-1.0.4.jar')"


JCMD=java
if [ -f "$JAVA_HOME/bin/java" ]
then
  echo "JAVA HOME: $JAVA_HOME"
  JCMD="$JAVA_HOME/bin/java"
fi

port_no='8081'
host_ip="$(hostname).$(dnsdomainname)"
#jmx_string="-Dcom.sun.management.jmxremote.port=${port_no} -Dcom.sun.management.jmxremote.authenticate=false -Dcom.sun.management.jmxremote.ssl=false -Djava.rmi.server.hostname=${host_ip}"

$JCMD -version

# check options
MEMORY=512m
MAIN=moa.gui.GUI
# launch class
#  -Dcom.sun.management.jmxremote.port="$port_no" -Dcom.sun.management.jmxremote.authenticate=false -Dcom.sun.management.jmxremote.ssl=false \
"$JCMD" \
  -classpath "$CLASSPATH" \
  -Xmx32g -Xms50m -Xss1g \
  -javaagent:"$JAVA_AGENT_PATH" \
  $MAIN
