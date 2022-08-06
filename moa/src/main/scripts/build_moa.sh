print_usage()
{
  echo "Usage: $0 <local_maven_repo> <maven_conda_env>"
  echo "e.g:   $0 ~/Desktop/m2_cache/ <maven_conda_env>"
  echo "e.g:   $0 /Scratch/ng98/JavaSetup1/local_m2/ /Scratch/ng98/JavaSetup1/conda"
}

if [ $# -lt 1 ]; then
  print_usage
  exit 1
else
  mavan_local_repo="$1"
fi

case $(uname)  in

  Darwin)
    echo "MacOS"
    export JAVA_HOME=`/usr/libexec/java_home -v 1.8.0_271`
    ;;

  Linux)
    echo "Linux"
    export JAVA_HOME='/usr/lib/jvm/java-8-oracle'
    ;;

  *)
    ;;
esac

if [ $# -gt 1 ]; then
  eval "$(conda shell.bash hook)"
  conda init bash
  conda activate "$2"
  conda env list
fi

if ! command -v mvn &> /dev/null
then
    echo "mvn could not be found"
    print_usage
    exit 1
fi

export MAVEN_OPTS="-Dmaven.repo.local=${mavan_local_repo}"
echo "maven info: ========================================"
mvn -v
echo "Building moa:======================================="
mvn clean install -DskipTests=true -Dmaven.javadoc.skip=true -Dlatex.skipBuild=true
echo""
echo "To run MOA GUI with NN support, execute command:======================================="
echo "bash moa/src/main/scripts/moa_gui_with_NN_support.sh <mavan_repo> <conda_env> <djl_cache_dir>"
echo "bash moa/src/main/scripts/moa_gui_with_NN_support.sh $mavan_local_repo $2 ~/Desktop/djl.ai"

