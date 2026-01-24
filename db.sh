#!/usr/bin/env bash

# Licensed to the LF AI & Data foundation under one
# or more contributor license agreements. See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership. The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

check_docker() {
    if ! docker info >/dev/null 2>&1; then
        echo "❌ Docker is not running."
        if [[ "$OSTYPE" == "darwin"* ]]; then
            echo "🚀 Attempting to start Docker Desktop..."
            open -a Docker
            echo "⏳ Waiting for Docker to start (this may take a minute)..."
            until docker info >/dev/null 2>&1; do
                sleep 5
            done
            echo "✅ Docker is now running."
        else
            echo "Please start Docker Desktop manually before running this script."
            exit 1
        fi
    fi
}

run_embed() {
    cat << EOF > embedEtcd.yaml
listen-client-urls: http://0.0.0.0:2379
advertise-client-urls: http://0.0.0.0:2379
quota-backend-bytes: 4294967296
auto-compaction-mode: revision
auto-compaction-retention: '1000'
EOF

# Need to change the two -v lines for it to work (remove pwd in both lines)
    sudo docker run -d \
        --name milvus-standalone \
        --security-opt seccomp:unconfined \
        -e ETCD_USE_EMBED=true \
        -e ETCD_DATA_DIR=/var/lib/milvus/etcd \
        -e ETCD_CONFIG_PATH=/milvus/configs/embedEtcd.yaml \
        -e COMMON_STORAGETYPE=local \
        -v milvus-volume:/var/lib/milvus \
        -v "$PWD/embedEtcd.yaml":/milvus/configs/embedEtcd.yaml \
        -p 19530:19530 \
        -p 9091:9091 \
        -p 2379:2379 \
        --health-cmd="curl -f http://localhost:9091/healthz" \
        --health-interval=30s \
        --health-start-period=90s \
        --health-timeout=20s \
        --health-retries=3 \
        milvusdb/milvus:v2.4.0 \
        milvus run standalone  1> /dev/null
}

wait_for_milvus_running() {
    echo "Wait for Milvus Starting..."
    while true
    do
        res=`sudo docker ps|grep milvus-standalone|grep healthy|wc -l`
        if [ $res -eq 1 ]
        then
            echo "Start successfully."
            break
        fi
        sleep 1
    done
}

start() {
    res=`sudo docker ps|grep milvus-standalone|grep healthy|wc -l`
    if [ $res -eq 1 ]
    then
        echo "Milvus is running."
        exit 0
    fi

    res=`sudo docker ps -a|grep milvus-standalone|wc -l`
    if [ $res -eq 1 ]
    then
        sudo docker start milvus-standalone 1> /dev/null
    else
        run_embed
    fi

    if [ $? -ne 0 ]
    then
        echo "Start failed."
        exit 1
    fi

    wait_for_milvus_running
}

start_attu() {
    res=`sudo docker ps|grep milvus-attu|wc -l`
    if [ $res -eq 1 ]
    then
        echo "Attu is running."
        exit 0
    fi

    res=`sudo docker ps -a|grep milvus-attu|wc -l`
    if [ $res -eq 1 ]
    then
        sudo docker start milvus-attu 1> /dev/null
    else
        sudo docker run -d \
            --name milvus-attu \
            --platform linux/amd64 \
            -p 3000:3000 \
            -e MILVUS_URL=127.0.0.1:19530 \
            zilliz/attu:v2.4.0 1> /dev/null
    fi
    echo "Attu started successfully. Access it at http://localhost:3000"
}

start_mongo() {
    # Check if port 27017 is already in use by a local process (like Homebrew)
    if lsof -Pi :27017 -sTCP:LISTEN -t >/dev/null ; then
        echo "Port 27017 is already in use (likely by Homebrew MongoDB). Skipping Docker MongoDB."
        return 0
    fi

    res=`sudo docker ps|grep mongo-rag|wc -l`
    if [ $res -eq 1 ]
    then
        echo "MongoDB container is already running."
        return 0
    fi

    res=`sudo docker ps -a|grep mongo-rag|wc -l`
    if [ $res -eq 1 ]
    then
        sudo docker start mongo-rag 1> /dev/null
    else
        sudo docker run -d \
            --name mongo-rag \
            -p 27017:27017 \
            mongo:latest 1> /dev/null
    fi
    echo "MongoDB started successfully at localhost:27017"
}

stop() {
    sudo docker stop milvus-standalone 1> /dev/null

    if [ $? -ne 0 ]
    then
        echo "Stop failed."
        exit 1
    fi
    echo "Stop successfully."
}

stop_attu() {
    sudo docker stop milvus-attu 1> /dev/null
    echo "Attu stopped successfully."
}

stop_mongo() {
    sudo docker stop mongo-rag 1> /dev/null
    echo "MongoDB stopped successfully."
}

# Delete containers
delete() {
    res=`sudo docker ps|grep -E "milvus-standalone|milvus-attu|mongo-rag"|wc -l`
    if [ $res -ge 1 ]
    then
        echo "Please stop Milvus, Attu, and MongoDB services before delete."
        exit 1
    fi
    sudo docker rm milvus-standalone 1> /dev/null 2>&1
    sudo docker rm milvus-attu 1> /dev/null 2>&1
    sudo docker rm mongo-rag 1> /dev/null 2>&1
    
    # Remove the local config file
    rm -f "$PWD/embedEtcd.yaml"
    
    echo "Containers removed. Note: Docker volumes (milvus-volume) were not deleted."
    echo "To delete data, run: sudo docker volume rm milvus-volume"
}

case $1 in
    start)
        check_docker
        start
        ;;
    stop)
        stop
        ;;
    start_attu)
        check_docker
        start_attu
        ;;
    stop_attu)
        stop_attu
        ;;
    start_mongo)
        check_docker
        start_mongo
        ;;
    stop_mongo)
        stop_mongo
        ;;
    start_all)
        check_docker
        start
        start_attu
        start_mongo
        ;;
    stop_all)
        stop_attu
        stop
        stop_mongo
        ;;
    delete)
        delete
        ;;
    *)
        echo "please use bash db.sh start|stop|start_attu|stop_attu|start_mongo|stop_mongo|start_all|stop_all|delete"
        ;;
esac