// ---------------------------------------------------------------------
//
// Copyright (C) 2020 - 2022 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE.md at
// the top level directory of deal.II.
//
// ---------------------------------------------------------------------

// Test ConsensusAlgorithms::selection().

#include <mpi.h>
#include <iostream>
#include <vector>

void run(const int target,
         const int reply,
         const bool use_iprobe,
         const MPI_Comm &comm)
{
  // MPI tags for sending requests and answers.
  const int tag_request = 1;
  const int tag_deliver = 2;

  // Answer buffer and request. Need to be outside scope to stay alive.
  std::vector<char> answer_buffer;
  MPI_Request answer_request;

  // 1) Send a request
  // to simplify this message has no payload
  // (the production code has one)
  std::vector<char> send_buffer;
  MPI_Request send_request;
  MPI_Isend(send_buffer.data(),
            send_buffer.size(),
            MPI_CHAR,
            target,
            tag_request,
            comm,
            &send_request);

  // 2) Until we receive a reply, answer requests and check for replies.
  int received_reply = 0;
  while (received_reply == 0)
  {
    // 2a) Check if there is a request from another process pending.
    {
      MPI_Status status;

      int request_is_pending = 0;
      if (use_iprobe == true)
      {
        // we may or may not have received a request, thats ok,
        // if we receive our reply and exit this loop before we 
        // get the request, we will answer the reply below,
        // after the IBarrier.
        MPI_Iprobe(reply,
                  tag_request,
                  comm,
                  &request_is_pending,
                  &status);
      }
      else
      {
        // probe only returns once request is received
        MPI_Probe(reply,
                  tag_request,
                  comm,
                  &status);
        request_is_pending = 1;
      }

      if (request_is_pending != 0)
      {
        // A request is pending. Receive and send reply.
        const auto other_rank = status.MPI_SOURCE;
        int message_size = 0;
        MPI_Get_count(&status, MPI_CHAR, &message_size);

        std::vector<char> buffer_recv(message_size);
        MPI_Recv(buffer_recv.data(),
                 buffer_recv.size(),
                 MPI_CHAR,
                 other_rank,
                 tag_request,
                 comm,
                 MPI_STATUS_IGNORE);

        MPI_Isend(answer_buffer.data(),
                  answer_buffer.size(),
                  MPI_CHAR,
                  other_rank,
                  tag_deliver,
                  comm,
                  &answer_request);
      }
    }
    // 2b) Check if we got a reply to our request
    {
      MPI_Status status;
      
      if (use_iprobe == true)
      {
        // we may or may not have a reply yet, thats ok, we will repeat
        // the loop until we get a reply.
        MPI_Iprobe(target,
                  tag_deliver,
                  comm,
                  &received_reply,
                  &status);
      }
      else
      {
        // Probe only returns when we got a reply
        MPI_Probe(target,
                  tag_deliver,
                  comm,
                  &status);
        received_reply = 1;
      }

      if (received_reply != 0)
      {
        // OK, so we have gotten a reply to our request from
        // one rank. Let us process it:
        const auto target = status.MPI_SOURCE;
        int message_size = 0;
        MPI_Get_count(&status, MPI_CHAR, &message_size);

        // message_size should always be 0 here.
        std::vector<char> recv_buffer(message_size);
        MPI_Recv(recv_buffer.data(),
                 recv_buffer.size(),
                 MPI_CHAR,
                 target,
                 tag_deliver,
                 comm,
                 MPI_STATUS_IGNORE);
      }
    }
  }

  // 3) Signal to all other processes that all requests of this
  //    process have been answered
  MPI_Request barrier_request;
  MPI_Ibarrier(comm, &barrier_request);

  // 4) Nevertheless, this process has to keep on answering
  //    (potential) incoming requests until all processes have
  //    received the answer to all requests
  int all_ranks_reached_barrier = 0;
  while (all_ranks_reached_barrier == 0)
  {
    {
      // Check if there is a request pending.
      MPI_Status status;

      int request_is_pending = 0;
      if (use_iprobe == true)
      {
        MPI_Iprobe(reply,
                    tag_request,
                    comm,
                    &request_is_pending,
                    &status);
      }
      // if we used probe above, we have already answered
      // the request that was sent to us, leave request_is_pending
      // at 0 and wait for Ibarrier to complete on all ranks.

      if (request_is_pending != 0)
      {
        // Get the rank of the requesting process.
        const auto other_rank = status.MPI_SOURCE;

        // get size of incoming message
        int message_size;
        MPI_Get_count(&status, MPI_CHAR, &message_size);

        // allocate memory for incoming message
        std::vector<char> buffer_recv(message_size);
        MPI_Recv(buffer_recv.data(),
                 buffer_recv.size(),
                 MPI_CHAR,
                 other_rank,
                 tag_request,
                 comm,
                 MPI_STATUS_IGNORE);

        MPI_Isend(answer_buffer.data(),
                  answer_buffer.size(),
                  MPI_CHAR,
                  other_rank,
                  tag_deliver,
                  comm,
                  &answer_request);
      }
    }

    // check if IBarrier has been reached by every rank
    MPI_Test(&barrier_request,
             &all_ranks_reached_barrier,
             MPI_STATUS_IGNORE);
  }

  // 5) wait for all requests to complete
  MPI_Wait(&send_request, MPI_STATUS_IGNORE);
  MPI_Wait(&barrier_request, MPI_STATUS_IGNORE);
  MPI_Wait(&answer_request, MPI_STATUS_IGNORE);

  return;
}

int main(int argc, char *argv[])
{
  int provided;
  int wanted = MPI_THREAD_SERIALIZED;
  MPI_Init_thread(&argc, &argv, wanted, &provided);

  const MPI_Comm comm = MPI_COMM_WORLD;

  int my_rank = 0;
  MPI_Comm_rank(comm, &my_rank);
  int n_rank = 1;
  MPI_Comm_size(comm, &n_rank);

  // send a message to the next higher rank, receive one from next lower
  const int target = (my_rank + 1) % n_rank;
  const int reply = (my_rank > 0) ? (my_rank - 1) : (n_rank - 1);

  // create a fresh MPI communicator for each part of the program
  MPI_Comm comm_workaround_1;
  MPI_Comm_dup(comm, &comm_workaround_1);

  // Using Probe instead of IProbe works
  if (my_rank == 0)
    std::cout << "Using blocking Probe:" << std::endl;

  for (unsigned int i = 0; i < 30000000; ++i)
  {
    if (i % 1000000 == 0 && my_rank == 0)
      std::cout << i << std::endl;

    const bool use_iprobe = false;
    run(target, reply, use_iprobe, comm_workaround_1);
  }

  // free communicator
  MPI_Comm_free(&comm_workaround_1);

  // Recreating the communicator for every iteration works,
  // but is very slow (at least until more than 30 million calls).
  if (my_rank == 0)
    std::cout << "Recreating communicator in each iteration with IProbe:" << std::endl;

  for (unsigned int i = 0; i < 30000000; ++i)
  {
    if (i % 1000000 == 0 && my_rank == 0)
      std::cout << i << std::endl;

    const bool use_iprobe = true;
    MPI_Comm comm_workaround_2;
    MPI_Comm_dup(comm, &comm_workaround_2);
    run(target, reply, use_iprobe, comm_workaround_2);
    MPI_Comm_free(&comm_workaround_2);
  }

  // create a fresh MPI communicator for each part of the program
  MPI_Comm comm_bugged;
  MPI_Comm_dup(comm, &comm_bugged);

  // This deadlocks after 16.7 million calls
  if (my_rank == 0)
    std::cout << "Using IProbe in same communicator:" << std::endl;

  for (unsigned int i = 0; i < 30000000; ++i)
  {
    if (i % 1000000 == 0 && my_rank == 0)
      std::cout << i << std::endl;

    const bool use_iprobe = true;
    run(target, reply, use_iprobe, comm_bugged);
  }

  // free communicator
  MPI_Comm_free(&comm_bugged);
}
