#!/usr/bin/python3

from collections import namedtuple
import re
import argparse
# Some helpful constant values that we'll be using.
Constants = namedtuple("Constants",["NUM_REGS", "MEM_SIZE", "REG_SIZE"])
constants = Constants(NUM_REGS = 8,
                      MEM_SIZE = 2**13,
                      REG_SIZE = 2**16)

class Subrow: # subrow === node
    def __init__(self): 
        self.vbit = 0
        self.tag = -1 
        self.next = None
        self.prev = None
    # getters
    def get_vbit(self):
        return self.vbit
    def get_tag(self):
        return self.tag
    # setters
    def set_vbit(self,new_vbit):
        self.vbit = new_vbit
    def set_tag(self,new_tag):
        self.tag = new_tag
class DoublyLinkedList:
    def __init__(self):
        self.head = None
        self.tail = None
    def append(self,node):
        if self.head is None or self.tail is None: # DLL is empty
            self.head = node
            self.tail = node
        else:
            self.tail.next = node
            node.prev = self.tail
            self.tail = node
    def remove(self, node):
        """ A special type of remove function that only affects middle of DLL and the tail"""
        if node == self.head: # we don't want to affect the head
            pass
        elif node == self.tail: # removing the tail
            self.tail = node.prev
            self.tail.next = None
            node.prev = None
        else: # removing from the middle
            node_before = node.prev
            node_after = node.next
            node_before.next = node_after
            node_after.prev = node_before
    def insert_at_beginning(self,node):
        """ A special type of insert that only inserts to the front of the list"""
        if node != self.head:
            node.next = self.head
            self.head.prev = node
            self.head = node
        # if node is head, don't do anything
def print_cache_config(cache_name, size, assoc, blocksize, num_rows):
    """
    Prints out the correctly-formatted configuration of a cache.

    cache_name -- The name of the cache. "L1" or "L2"

    size -- The total size of the cache, measured in memory cells.
        Excludes metadata

    assoc -- The associativity of the cache. One of [1,2,4,8,16]

    blocksize -- The blocksize of the cache. One of [1,2,4,8,16,32,64])

    num_rows -- The number of rows in the given cache.

    sig: str, int, int, int, int -> NoneType
    """

    summary = "Cache %s has size %s, associativity %s, " \
        "blocksize %s, rows %s" % (cache_name,
        size, assoc, blocksize, num_rows)
    print(summary)
def print_log_entry(cache_name, status, pc, addr, row):
    """
    Prints out a correctly-formatted log entry.

    cache_name -- The name of the cache where the event
        occurred. "L1" or "L2"

    status -- The kind of cache event. "SW", "HIT", or
        "MISS"

    pc -- The program counter of the memory
        access instruction

    addr -- The memory address being accessed.

    row -- The cache row or set number where the data
        is stored.

    sig: str, str, int, int, int -> NoneType
    """
    log_entry = "{event:8s} pc:{pc:5d}\taddr:{addr:5d}\t" \
        "row:{row:4d}".format(row=row, pc=pc, addr=addr,
            event = cache_name + " " + status)
    print(log_entry)
def load_machine_code(machine_code, mem):
    """
    Loads an E20 machine code file into the list
    provided by mem. We assume that mem is
    large enough to hold the values in the machine
    code file.
    sig: list(str) -> list(int) -> NoneType
    """
    machine_code_re = re.compile("^ram\[(\d+)\] = 16'b(\d+);.*$")
    expectedaddr = 0
    for line in machine_code:
        match = machine_code_re.match(line)
        if not match:
            raise ValueError("Can't parse line: %s" % line)
        addr, instr = match.groups()
        addr = int(addr,10)
        instr = int(instr,2)
        if addr != expectedaddr:
            raise ValueError("Memory addresses encountered out of sequence: %s" % addr)
        if addr >= len(mem):
            raise ValueError("Program too big for memory")
        expectedaddr += 1
        mem[addr] = instr
def revealTrueimm7(imm7):
    """
    the imm7's binary is signed but in imm7 Python variable, it's always saved as a positive decimal value
    this function reveals its "true" decimal value <--- take imm7's sign into consideration
    sig: int --> int
    """
    if((0 <= imm7) and (imm7 <= 63)): # means imm7 in binary is 0000000 to 0111111 <--> imm7's MSB is 0  <--> imm7 is positive --> return imm7 as is
        return imm7
    else: # imm7 >= 64 <--> means imm7 in binary is 1000000 to 1111111 <--> imm7 in binary's MSB is 1 <--> imm7 is negative 
          # --> we get its true decimal value by grabbing from the LSB up to the (n-1)th bit and then subtract the nth bit
        return (imm7 & 63) - 64     # &63 grabs from LSB up to the (n-1)th bit
                                    # -64 is subtracting the nth bit
def revealTrueRelimm(relimm):
    """
    reveal the "true" decimal value of the relimm7 that's passed in
    sig: int --> int
    """
    if((0 <= relimm) and (relimm <= 63)):
        return relimm
    else:
        return (relimm & 32767) - 32768 # grab from LSB up to idx14
                                        # minus off idx15
def signExtend(imm7):
    """
    the imm7 argument is signed
    this function will extend imm7 to 16-bits using 1s if imm7's MSB is 1 
    this function will extend imm7 to 16-bits using 0s if imm7's MSB is 0 === do nothing if imm7's MSB is 0 
    sig: int --> int
    """

    if( (0 <= imm7) and (imm7 <= 63)): # means imm7 in binary is 0000000 to 0111111 <--> imm7's MSB is 0 <--> imm7 is positive --> sign extend by zero-extending --> return imm7 as is
        return imm7
    else: # means imm7 in binary is 1000000 to 1111111 <--> imm7's MSB is 1 <---> imm7 is negative --> sign-extend by extending with 1's
        return (imm7 | 65408)   # our mask should have idx7 to idx15 as 1s and idx0 to idx6 as 0
                                # we should operate with OR so idx7 through idx15 get turned on to 1
def getMemoryAddress(pc):
    """
    This function will check if the memory cell address is within the available memory cells 0 through 8191
    If yes --> no modifications are needed for the address
    If no --> modifications are needed for the address
    sig: int --> int
    """
    if((0 <= pc) and (pc <= 8191)): # address is within the available 0 to 8191 memory cell addresses
        return pc # address is valid as is --> no need to modify address --> return address as is
    else: # address is either overflowed or underflowed --> handle both cases in the same manner
          # we want pc to wrap around to the other side of the range
        return (pc % 8192) # mod to handle wrapping
def simulation(pc, registers, memory, cache_config, log):
    """
    This function runs the simulation of E20
    sig: int, list, list --> int
    """
    """ obtain the first line of instructions at pc == 0 aka memory cell 0 """
    instructions = memory[pc]     # pc == 0
                                  # instructions are the contents of each memory cell
    opcode = (instructions & 57344) >> 13    # want to capture bits 15 14 13 to get opcode so we isolate them and then shift
    rA   = (instructions & 7168  )  >> 10       # want to capture bits 12 11 10 to get rA so we isolate them and then shift
    rB   = (instructions & 896 ) >> 7       # want to capture bits 9 8 7 to get rB so we isolate them and then shift
    rC   = (instructions & 112)  >> 4      # want to capture bits 6 5 4 to get rC so isolate them and then shift
    imm13 = instructions & 8191           # isolate bits 12 11 ... 0 to get imm13
    imm7 = instructions &  127            # isolate bits 6 5 4 ... 0 to get imm7
    func = instructions &  15             # isolate bits 3 2 1 0 to get func
                                          # func is used to distinguish between operations that have the same opcode, namely the opcode 000

                                          # in Python, all of these variables are treated as DECIMAL numbers
    """initialize our cache"""
    if(len(cache_config) == 3): # we are doing single cache simulation
        L1cacheSize = cache_config[0]
        L1assoc = cache_config[1]
        L1blockSize = cache_config[2]
        L1totalRows = L1cacheSize//(L1assoc*L1blockSize) # total rows = cacheSize // (assoc * blockSize)
        L1 = build_cache(L1totalRows, L1assoc) # L1 cache
        L2 = None
    else: # we are doing double cache simulation
        L1cacheSize = cache_config[0]
        L1assoc = cache_config[1]
        L1blockSize = cache_config[2]
        L1totalRows = L1cacheSize//(L1assoc*L1blockSize) # total rows = cacheSize // (assoc * blockSize)
        L1 = build_cache(L1totalRows, L1assoc) # L1 cache
        L2cacheSize = cache_config[3]
        L2assoc = cache_config[4]
        L2blockSize = cache_config[5]
        L2totalRows = L2cacheSize//(L2assoc*L2blockSize) # total rows = cacheSize // (assoc * blockSize)
        L2 = build_cache(L2totalRows, L2assoc) # L2 cache
    """ the core of our simulation """
    while( not (opcode == 2 and imm13 == pc) ):         # while not halt, keep reading, parsing and executing instructions line-by-line (ie cell-by-cell)
                                                        # halt === opcode == 2 in decimal AND jumping in place which means jumping to current pc

        if (opcode == 0): # operations with three register arguments: add, sub, or, and, slt, jr 
                          # we need to further distinguish these operations by their func bits
            if(func == 8): # operation is jr
                pc = registers[rA]
            else: # possible operations are add, sub, or, and, slt
                if (rC == 0): # if we try to modify register0 by storing value in register 0 --> skip this instruction 
                    pass
                else: # our dst register is not $0
                    # obtain the values within our registers
                    srcA_value = registers[rA]
                    srcB_value = registers[rB]
                    if(func == 0): # operation is add
                        sum = srcA_value + srcB_value 
                        if (sum > 65535): # 65535 is the largest decimal number that the 16-bit register can hold            
                            sum = (sum & 65535)  # handling overflow --> we need to wrap around to other end of the 16-bit range
                        registers[rC] = sum
                    elif (func == 1): # operation is sub
                        difference = srcA_value - srcB_value
                        if (difference < 0): # handling underflow --> we need to wrap around to other end of the 16-bit range
                            difference = (difference & 65535) 
                        registers[rC] = difference
                    elif (func == 2): # operation is or
                        registers[rC] = (srcA_value | srcB_value)
                    elif(func== 3): # operation is and
                        registers[rC] = (srcA_value & srcB_value)
                    else: # operation is slt
                        if (srcA_value < srcB_value): # comparison is done unsigned
                            registers[rC] = 1
                        else:
                            registers[rC] = 0
                pc = pc + 1 # these instructions proceed pc forward by 1
        # operations for two register arguments 
        elif(opcode == 7): # operation is slti
            if(rB == 0): # if we try to modify register0 by storing value in register 0 --> skip the instruction
                pass
            else: # we're not trying to modify register0
                srcA_value = registers[rA]
                imm7 = signExtend(imm7) # imm7 is sign-extended as per manual
                if(srcA_value < imm7): # comparison is done unsigned
                    registers[rB] = 1
                else:
                    registers[rB] = 0
            pc = pc + 1 
        elif(opcode == 4): # operation is load word
            srcA_value = registers[rA]
            imm7 = revealTrueimm7(imm7)  # imm7 is signed
            memoryIdx = getMemoryAddress(srcA_value + imm7) # we want the contents from this memory address
            if (L2 == None): # single cache simulation
                blockID = memoryIdx // L1blockSize
                targetRowIdx = blockID % L1totalRows
                targetTag = blockID // L1totalRows
                single_cache("LW", [targetRowIdx,targetTag], [pc, memoryIdx], L1,log)
            else: # double cache simulation
                blockID1 = memoryIdx // L1blockSize
                targetRowIdx1 = blockID1 % L1totalRows
                targetTag1 = blockID1 // L1totalRows
                blockID2 = memoryIdx // L2blockSize
                targetRowIdx2 = blockID2 % L2totalRows
                targetTag2 = blockID2 // L2totalRows
                double_cache("LW", [targetRowIdx1,targetTag1,targetRowIdx2,targetTag2], [pc, memoryIdx], [L1,L2],log)
            if rB != 0: # only modify register when register is not reg0
                registers[rB] = memory[memoryIdx] # put the contents of the memory address into the destination register
            pc = pc + 1
        elif(opcode == 5): # operation is save word
            srcA_value = registers[rA]
            imm7 = revealTrueimm7(imm7) # imm7 is signed
            memoryIdx = getMemoryAddress(srcA_value + imm7) # this is the memory address we want to place register's value into
            if (L2 == None): # single cache simulation
                    blockID = memoryIdx // L1blockSize
                    targetRowIdx = blockID % L1totalRows
                    targetTag = blockID // L1totalRows
                    single_cache("SW", [targetRowIdx,targetTag], [pc, memoryIdx], L1,log)
            else: # double cache simulation
                    blockID1 = memoryIdx // L1blockSize
                    targetRowIdx1 = blockID1 % L1totalRows
                    targetTag1 = blockID1 // L1totalRows
                    blockID2 = memoryIdx // L2blockSize
                    targetRowIdx2 = blockID2 % L2totalRows
                    targetTag2 = blockID2 // L2totalRows
                    double_cache("SW", [targetRowIdx1,targetTag1,targetRowIdx2,targetTag2], [pc, memoryIdx], [L1,L2], log)
            memory[memoryIdx] = registers[rB] # put the value of registerB into the cell of target memory address
            pc = pc + 1
        elif(opcode == 6): # operation is jeq
            srcA_value = registers[rA]
            srcB_value = registers[rB]
            if(srcA_value == srcB_value): # srcA_value == srcB_value --> we should jump
                pc = revealTrueRelimm(signExtend(imm7)) + pc + 1    # when a jump is performed --> rel_imm is sign extended and added to the successor value of the program counter
                                                                                        # rel_imm is the 7bits we grabbed from instructions
                                                                                        # successor value of the program counter is pc+1
                                                                                        # imm7 is signed
            else: # srcA_value =/= srcB_value
                pc = pc + 1
        elif(opcode == 1): # operation is addi
            if (rB == 0): # if we try to modify register0 by storing value in register 0 --> skip the instruction
                pass
            else: # we're not trying to modify register0
                srcA_value = registers[rA]
                imm7 = revealTrueimm7(imm7) # imm7 is signed
                sum = srcA_value + imm7
                if((sum > 65535) or (sum < 0)): # handle overflow / underflow of the sum
                                                # sum can be negative if we add a negative immediate value
                    sum = (sum & 65535)
                registers[rB] = sum
            pc = pc + 1
        # operations with no register arguments 
        elif (opcode == 2): # operation is j 
            pc = imm13 # jump to the given immediate value address
        elif(opcode == 3): # operation is jal
            registers[7] = pc + 1 # put the address of the subsequent instruction (aka memory cell) into register7
                                                 # this saved address must be adjusted 
            pc = imm13 # jump to the given immediate value address
        else: # if the machine code has invalid / undefined instructions then we ignore
            pc = pc+1     
        # obtain the next line (aka next cell) of instructions
        instructions = memory[getMemoryAddress(pc)]
        opcode = (instructions & 57344) >> 13    # want to capture bits 15 14 13 to get opcode so we isolate them and then shift
        rA   = (instructions & 7168  )  >> 10       # want to capture bits 12 11 10 to get rA so we isolate them and then shift
        rB   = (instructions & 896 ) >> 7       # want to capture bits 9 8 7 to get rB so we isolate them and then shift
        rC   = (instructions & 112)  >> 4      # want to capture bits 6 5 4 to get rC so we isolate them and then shift
        imm13 = (instructions & 8191)         # isolate bits 12 11 ... 0 to get imm13
        imm7 = (instructions &  127)          # isolate bits 6 5 4 ... 0 to get imm7
        func = (instructions &  15)           # isolate bits 3 2 1 0 to get func
                                              # func is used to distinguish between operations that have the same opcode, namely the opcode 000

                                              # in Python, all of these variables are treated as DECIMAL numbers
    # out of while loop means we've reached halt === our E20 program has concluded --> componenents (pc, registers, memory) should be updated
    return pc
def build_cache(row_count, assoc):
    """
    This function initializes an empty cache
    sig: int, int ---> list of DLLs
    """
    cache = []
    for i in range(row_count):
        row = DoublyLinkedList()
        for j in range(assoc): # linking all of the nodes (ie subrows) together to create DLL
            new_subrow = Subrow()
            row.append(new_subrow)
        cache.append(row) # put the DLL in the cache
    return cache # a list of DLL
def update_log(l1_activityLst, l2_activityLst, log):
    if l2_activityLst == None: # only one entry needs to be created 
        cache_name = l1_activityLst[0]
        cache_outcome = l1_activityLst[1]
        pc = l1_activityLst[2]
        address = l1_activityLst[3]
        row = l1_activityLst[4]
        entry = [cache_name, cache_outcome, pc, address, row] # create a log entry
        log.append(entry) # put entry into log
    else: # two entries needed to be created
        cache1_name = l1_activityLst[0]
        cache1_outcome = l1_activityLst[1]
        pc = l1_activityLst[2]
        address = l1_activityLst[3]
        cache1_row = l1_activityLst[4]
        cache2_name = l2_activityLst[0]
        cache2_outcome = l2_activityLst[1]
        cache2_row = l2_activityLst[4]
        entry1 = [cache1_name, cache1_outcome, pc, address, cache1_row]
        entry2 = [cache2_name, cache2_outcome, pc, address, cache2_row]
        log.append(entry1)
        log.append(entry2)
def single_cache(action, targetsLst, simu_dataLst, cache,log): # simulate single cache activities
    outcome = None 
    target_row_idx = targetsLst[0]
    target_tag = targetsLst[1]
    target_row = cache[target_row_idx] # target_row is a DLL object
    curr_subrow = target_row.head
    while(curr_subrow and curr_subrow.get_vbit() == 1):  # use the fact that all of the 1 vbits will be at the front of the list and 0 vbits will be at the end of the list
        if(curr_subrow.get_tag() == target_tag): # we got a hit --> a match between stored tag and target tag
                if action == "LW":
                    outcome = "HIT" # log the result
                else:
                    outcome = "SW"
                # update usage recency
                target_row.remove(curr_subrow)
                target_row.insert_at_beginning(curr_subrow)
                break # we are done so break out of the this loop
        curr_subrow = curr_subrow.next
    if(outcome == None): # we've exited the traversal with outcome not set which means we've missed
        if(curr_subrow): # means we exited the traversal AND node still exists
                            # means we exited the traversal because of meeting a 0 vbit --> there's still availability in the row --> no eviction required
            # update the subrow
            curr_subrow.set_vbit(1)
            curr_subrow.set_tag(target_tag)
            # update the usage recency
            target_row.remove(curr_subrow)
            target_row.insert_at_beginning(curr_subrow) 
        else: # we've exited the traversal AND node no longer exists --> all of the subrows have vbits of 1 and we still didn't get a match --> row is full --> eviction is needed
            # tail will always be least recently used ---> tail is always prone to eviction
            evicted_subrow = target_row.tail
            evicted_subrow.set_tag(target_tag)
            # update usage recency
            target_row.remove(evicted_subrow)
            target_row.insert_at_beginning(evicted_subrow)
        if action == "LW": 
            outcome = "MISS" # log the result
        else:
            outcome = "SW"
    l1_activityLst = ["L1",outcome,simu_dataLst[0],simu_dataLst[1],target_row_idx] # collect all relevant info
    l2_activityLst = None            
    update_log(l1_activityLst,l2_activityLst,log) # record cache activity in log
def double_cache(action, targetsLst, simu_dataLst, cacheLst, log): # simulate double cache activities
    l1_cache = cacheLst[0]
    l2_cache = cacheLst[1]
    l1_targetRowIdx = targetsLst[0]
    l1_targetTag =  targetsLst[1]
    l2_targetRowIdx = targetsLst[2]
    l2_targetTag = targetsLst[3]
    l1_outcome = None
    l2_outcome = None  
    l1_target_row = l1_cache[l1_targetRowIdx] # l1_target_row is a DLL object
    l1_curr_subrow = l1_target_row.head
    while(l1_curr_subrow and l1_curr_subrow.get_vbit() == 1): # use the fact that all 1 vbits are at the front and all 0 vbits are at the rear
                                                                # keep traversing until we hit a 0 vbit or exhaust list
        if(l1_targetTag == l1_curr_subrow.get_tag()): # this is the first situation where only L1 cache is consulted and L2 is ignored
            if action == "LW":
                l1_outcome = "HIT"
                l1_activityLst = ["L1", l1_outcome, simu_dataLst[0], simu_dataLst[1], l1_targetRowIdx]
                l2_activityLst = None
                l1_target_row.remove(l1_curr_subrow)
                l1_target_row.insert_at_beginning(l1_curr_subrow)
            else:
                l1_outcome = "SW"
            break
        l1_curr_subrow = l1_curr_subrow.next
    if(l1_outcome == None or l1_outcome == "SW"): # l1 missed --> we must consult l2
                                                  # l1 write hit ---> we write through to l2
        if action == "LW": # check if we read or write
            l1_outcome = "MISS"
        else:
            l1_outcome = "SW"
        # handle l2's activity
        l2_target_row = l2_cache[l2_targetRowIdx]
        l2_curr_subrow = l2_target_row.head
        while(l2_curr_subrow and l2_curr_subrow.get_vbit() == 1): # use the fact that all 1 vbits are at the front and all 0 vbits are at the rear
                                                                    # keep traversing until we hit a 0 vbit or exhaust list
            if(l2_targetTag == l2_curr_subrow.get_tag()): # this is second situation where L1 cache missed but L2 hit
                if action == "LW": # check if we read or write
                    l2_outcome = "HIT"
                else:
                    l2_outcome = "SW"
                l2_target_row.remove(l2_curr_subrow)
                l2_target_row.insert_at_beginning(l2_curr_subrow)
                break
            l2_curr_subrow = l2_curr_subrow.next
        if(l2_outcome == None):
            if action == "LW": # check if we read or write
                l2_outcome = "MISS"  # this is third situation where L1 missed and L2 missed
            else:
                l2_outcome = "SW"
            if(l2_curr_subrow): # we exited l2 traversal due to meeting a 0 vbit --> there's still availability in l2 --> no need to evict
                l2_curr_subrow.set_vbit(1)
                l2_curr_subrow.set_tag(l2_targetTag)
                l2_target_row.remove(l2_curr_subrow)
                l2_target_row.insert_at_beginning(l2_curr_subrow)
            else: # we exited l2 traversal due to exhaustion --> all vbits are 1 --> there's no availability in l2 --> need to evict
                l2_evicted_subrow = l2_target_row.tail # tail is always LRU --> prone to eviction
                l2_evicted_subrow.set_tag(l2_targetTag)
                l2_target_row.remove(l2_evicted_subrow)
                l2_target_row.insert_at_beginning(l2_evicted_subrow)
        # remember to update l1 cache usage recency
        # after l2 activity, regardless l2 hit or miss, l1 is updated
        # after l1 write, l1 must also be updated
        if(l1_curr_subrow):  # we exited the l1 traversal due to meeting a 0 vbit --> there's still availability in l1 --> no need to evict
            l1_curr_subrow.set_vbit(1)
            l1_curr_subrow.set_tag(l1_targetTag)
            l1_target_row.remove(l1_curr_subrow)
            l1_target_row.insert_at_beginning(l1_curr_subrow)
        else: # we exited l1 traversal due to exhaustion --> all vbits are 1 --> there's no availability in l1 --> need to evict
            l1_evicted_subrow = l1_target_row.tail # tail is always LRU --> prone to eviction
            l1_evicted_subrow.set_tag(l1_targetTag)
            l1_target_row.remove(l1_evicted_subrow)
            l1_target_row.insert_at_beginning(l1_evicted_subrow)
        l1_activityLst = ["L1", l1_outcome, simu_dataLst[0],simu_dataLst[1],l1_targetRowIdx]
        l2_activityLst = ["L2", l2_outcome, simu_dataLst[0],simu_dataLst[1],l2_targetRowIdx]
    update_log(l1_activityLst,l2_activityLst,log) # record cache activity in log
def main():
    parser = argparse.ArgumentParser(description='Simulate E20 cache')
    parser.add_argument('filename', help=
        'The file containing machine code, typically with .bin suffix')
    parser.add_argument('--cache', help=
        'Cache configuration: size,associativity,blocksize (for one cache) '
        'or size,associativity,blocksize,size,associativity,blocksize (for two caches)')
    cmdline = parser.parse_args()
    """ initial states of our simulator """
    pc = 0
    registers = []
    for i in range(8): # initialize all 8 registers to 0
        registers.append(0)
    memory = []
    for i in range(8192): # initialize all 8192 memory cells to 0
        memory.append(0)
    log = [] # initialize our cache log
    with open(cmdline.filename) as file:
        load_machine_code (file, memory)
    if cmdline.cache is not None:
        parts = cmdline.cache.split(",")
        if len(parts) == 3:
            [L1size, L1assoc, L1blocksize] = [int(x) for x in parts]
            # TODO: execute E20 program and simulate one cache here
            pc = simulation(pc, registers, memory, [L1size, L1assoc, L1blocksize],log) # simulate the processor
                                                                                   # also pass cache config into simulator
            print_cache_config("L1", L1size, L1assoc, L1blocksize, L1size//(L1assoc*L1blocksize)) # printing the cache configuration
            for i in range(len(log)):
                print_log_entry(log[i][0], log[i][1],log[i][2],log[i][3],log[i][4]) # [0] is cache name
                                                                                    # [1] is cache outcome/activity
                                                                                    # [2] is pc, which line of instruction we're on
                                                                                    # [3] is address, which memory address is affected
                                                                                    # [4] is which row in the cache is being affected
                                                                                    # i is the ith entry of the log
        elif len(parts) == 6:
            [L1size, L1assoc, L1blocksize, L2size, L2assoc, L2blocksize] = \
                [int(x) for x in parts]
            # TODO: execute E20 program and simulate two caches here
            pc = simulation(pc, registers, memory, [L1size, L1assoc, L1blocksize, L2size, L2assoc, L2blocksize], log) # simulate the processor
                                                                                                                  # also pass cache config into simulator
            print_cache_config("L1", L1size, L1assoc, L1blocksize, L1size//(L1assoc*L1blocksize)) # printing the cache configuration
            print_cache_config("L2", L2size, L2assoc, L2blocksize, L2size//(L2assoc*L2blocksize)) # printing the cache configuration
            for i in range(len(log)):
                print_log_entry(log[i][0], log[i][1],log[i][2],log[i][3],log[i][4]) # [0] is cache name
                                                                                    # [1] is cache event
                                                                                    # [2] is pc, which line of instruction we're on
                                                                                    # [3] is address, which memory address is affected
                                                                                    # [4] is which row in the cache is being affected
                                                                                    # i is the ith entry of the log
        else:
            raise Exception("Invalid cache config")
if __name__ == "__main__":
    main()
#ra0Eequ6ucie6Jei0koh6phishohm9
