#include "LinearTablePartitioner.h"
#include "BufferMgr.h"
#include "Buffer.h"
#include <math.h>
#include <iostream>

#include <assert.h>
#include <boost/lexical_cast.hpp>

using Buffer_Namespace::Buffer;
using Buffer_Namespace::BufferMgr;

using namespace std;



LinearTablePartitioner::LinearTablePartitioner(const int tableId,  vector <ColumnInfo> &columnInfoVec, Buffer_Namespace::BufferMgr &bufferManager, const mapd_size_t maxPartitionRows, const mapd_size_t pageSize /*default 1MB*/) :
		tableId_(tableId), bufferManager_(bufferManager), maxPartitionRows_(maxPartitionRows), pageSize_(pageSize), maxPartitionId_(-1), pgConnector_("mapd","tmostak")/*, currentInsertBufferSize_(0) */ {
    // populate map with ColumnInfo structs
    for (vector <ColumnInfo>::iterator colIt = columnInfoVec.begin(); colIt != columnInfoVec.end(); ++colIt) {
        columnMap_[colIt -> columnId_] = *colIt; 
    }
    createStateTableIfDne();
    readState();

    // need to query table here to see how many rows are in each partition
    //createNewPartition();
    /*
    PartitionInfo fragInfo;
    fragInfo.partitionId = 0;
    fragInfo.numTuples = 0;
    partitionInfoVec_
    */

}

LinearTablePartitioner::~LinearTablePartitioner() {
    writeState();
}

void LinearTablePartitioner::insertData (const vector <int> &columnIds, const vector <void *> &data, const int numRows) {
    if (maxPartitionId_ < 0 || partitionInfoVec_.back().numTuples_ + numRows > maxPartitionRows_) { // create new partition - note that this as currently coded will leave empty tuplesspace at end of current buffer chunks in the case of an insert of multiple rows at a time 
        // should we also do this if magPartitionId_ < 0 and allocate lazily?
        createNewPartition();
    }

    for (int c = 0; c != columnIds.size(); ++c) {
        int columnId = columnIds[c];
        map <int, ColumnInfo>::iterator colMapIt = columnMap_.find(columnId);
        // This SHOULD be found and this iterator should not be end()
        // as SemanticChecker should have already checked each column reference
        // for validity
        assert(colMapIt != columnMap_.end());
        //cout << "Insert buffer before insert: " << colMapIt -> second.insertBuffer_ << endl;
        //cout << "Insert buffer before insert length: " << colMapIt -> second.insertBuffer_ -> length() << endl;
        colMapIt -> second.insertBuffer_ -> append(colMapIt -> second.bitSize_ * numRows / 8, static_cast <mapd_addr_t> (data[c]));
    }
    //currentInsertBufferSize_ += numRows;
    partitionInfoVec_.back().numTuples_ += numRows;
}



void LinearTablePartitioner::createNewPartition() { 
    // also sets the new partition as the insertBuffer_ for each column

    // iterate through all ColumnInfo structs in map, unpin previous insert buffer and
    // create new insert buffer
    maxPartitionId_++;
    for (map<int, ColumnInfo>::iterator colMapIt = columnMap_.begin(); colMapIt != columnMap_.end(); ++colMapIt) {
        if (colMapIt -> second.insertBuffer_ != 0)
            colMapIt -> second.insertBuffer_ -> unpin();
        ChunkKey chunkKey = {tableId_, maxPartitionId_,  colMapIt -> second.columnId_};
        // We will allocate enough pages to hold the maximum number of rows of
        // this type
        mapd_size_t numPages = ceil(static_cast <float> (maxPartitionRows_) / (pageSize_ * 8 / colMapIt -> second.bitSize_)); // number of pages for a partition is celing maximum number of rows for a partition divided by how many elements of this column type can fit on a page 
        //cout << "NumPages: " << numPages << endl;
        colMapIt -> second.insertBuffer_ = bufferManager_.createChunk(chunkKey, numPages , pageSize_);
        //cout << "Insert buffer address after create: " << colMapIt -> second.insertBuffer_ << endl;
        //cout << "Length: " << colMapIt -> second.insertBuffer_ -> length () << endl;
        //if (!partitionInfoVec_.empty()) // if not first partition
        //    partitionInfoVec_.back().numTuples_ = currentInsertBufferSize_;
    }
    PartitionInfo newPartitionInfo;
    newPartitionInfo.partitionId_ = maxPartitionId_;
    newPartitionInfo.numTuples_ = 0; 
    partitionInfoVec_.push_back(newPartitionInfo);
}

void LinearTablePartitioner::getPartitionsForQuery(vector <PartitionInfo> &partitions, const void *predicate) {
    // right now we don't test predicate, so just return all partitions 
    partitions = partitionInfoVec_;

}

void LinearTablePartitioner::createStateTableIfDne() {
     mapd_err_t status = pgConnector_.query("CREATE TABLE IF NOT EXISTS fragments(table_id INT, fragment_id INT, num_rows INT, PRIMARY KEY (table_id, fragment_id))");
     assert(status == MAPD_SUCCESS);
}


void LinearTablePartitioner::readState() {
    string partitionQuery ("select fragment_id, num_rows from fragments where table_id = " + boost::lexical_cast <string> (tableId_));
    partitionQuery += " order by fragment_id";
    mapd_err_t status = pgConnector_.query(partitionQuery);
    assert(status == MAPD_SUCCESS);
    size_t numRows = pgConnector_.getNumRows();
    partitionInfoVec_.resize(numRows);
    for (int r = 0; r < numRows; ++r) {
        partitionInfoVec_[r].partitionId_ = pgConnector_.getData<int>(r,0);  
        partitionInfoVec_[r].numTuples_ = pgConnector_.getData<int>(r,1);
    }
    if (numRows > 0) {
        maxPartitionId_ = partitionInfoVec_[numRows-1].partitionId_; 
        getInsertBufferChunks();
    }
        // now need to get existing chunks
}

void LinearTablePartitioner::getInsertBufferChunks() {
    for (map<int, ColumnInfo>::iterator colMapIt = columnMap_.begin(); colMapIt != columnMap_.end(); ++colMapIt) {
        if (colMapIt -> second.insertBuffer_ != 0)
            colMapIt -> second.insertBuffer_ -> unpin();
        ChunkKey chunkKey = {tableId_, maxPartitionId_,  colMapIt -> second.columnId_};
        colMapIt -> second.insertBuffer_ = bufferManager_.getChunkBuffer(chunkKey);
        assert (colMapIt -> second.insertBuffer_ != NULL);
        //@todo change assert into throwing an exception
    }
}


void LinearTablePartitioner::writeState() {
    // do we want this to be fully durable or will allow ourselves
    // to delete existing rows for this table
    // out of convenience before adding the
    // newest state back in?
    // Will do latter for now as we do not 
    // plan on using postgres forever for metadata
     string deleteQuery ("delete from fragments where table_id = " + boost::lexical_cast <string> (tableId_));
     mapd_err_t status = pgConnector_.query(deleteQuery);
     assert(status == MAPD_SUCCESS);
    for (auto partIt = partitionInfoVec_.begin(); partIt != partitionInfoVec_.end(); ++partIt) {
        string insertQuery("INSERT INTO fragments (table_id, fragment_id, num_rows) VALUES (" + boost::lexical_cast<string>(tableId_) + "," + boost::lexical_cast<string>(partIt -> partitionId_) + "," + boost::lexical_cast<string>(partIt -> numTuples_) + ")"); 
        status = pgConnector_.query(insertQuery);
         assert(status == MAPD_SUCCESS);
    }
}
