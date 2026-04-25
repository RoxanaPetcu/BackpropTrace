# # Get the internal IP of this node
# export RETRIEVER_IP=$(hostname -I | awk '{for(i=1;i<=NF;i++) if($i !~ /^169\.254/) {print $i; exit}}')
# # Save it to scratch so the Trainer job can read it
# echo "$RETRIEVER_IP" > "$RETRIEVER_IP_PATH"
# echo "RETRIEVER IS RUNNING ON INTERNAL IP: $RETRIEVER_IP"

# file_path="$DATA_PATH"
# index_file=$file_path/e5_Flat.index
# corpus_file=$file_path/wiki-18.jsonl
# retriever_name=e5
# retriever_path=intfloat/e5-base-v2

# python backproptrace/search/retrieval_server.py --index_path $index_file \
#                                             --corpus_path $corpus_file \
#                                             --topk 3 \
#                                             --retriever_name $retriever_name \
#                                             --retriever_model $retriever_path \
#                                             --faiss_gpu


module purge
module load 2025
module load Anaconda3/2025.06-1

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate retriever

# Get the internal IP of this node
export RETRIEVER_IP=$(hostname -I | awk '{for(i=1;i<=NF;i++) if($i !~ /^169\.254/) {print $i; exit}}')
# Save it to scratch so the Trainer job can read it
echo "$RETRIEVER_IP" > /scratch-shared/rpetcu/current_retriever_ip_backproptrace.txt
echo "RETRIEVER IS RUNNING ON INTERNAL IP: $RETRIEVER_IP"

file_path=/scratch-shared/rpetcu/data/subsearch
index_file=$file_path/e5_Flat.index
corpus_file=$file_path/wiki-18.jsonl
retriever_name=e5
retriever_path=intfloat/e5-base-v2

python subsearch/search/retrieval_server.py --index_path $index_file \
                                            --corpus_path $corpus_file \
                                            --topk 3 \
                                            --retriever_name $retriever_name \
                                            --retriever_model $retriever_path \
                                            --faiss_gpu