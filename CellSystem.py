from Signal import NUM_SIGBITS, NUM_ACVIVATED, SIGNALS, POWERS

class FanoCell :
    def __init__(self):
        self.act_mask = 0b0
        self.pwr_mask = ((0b1 << NUM_SIGBITS) - 1) << NUM_SIGBITS
        self.sav_sig = 0b0
        self.act = False

    def setmask(self, mask):
        self.act_mask = mask

    def count(self, bits):
        n = 0
        first_mask = 0b1
        for i in range(NUM_SIGBITS*2) :
            if (bits & first_mask) > 0 :
                n += 1
            first_mask = first_mask << 1
        return n

    def compare(self, sig):
        self.act = self.count(self.act_mask & sig) >= NUM_ACVIVATED
        # print(
        #     "ACT mask :", ("{0:0"+str(NUM_SIGBITS)+"b}").format(self.act_mask),
        #     "| PWR mask :", ("{0:0"+str(NUM_SIGBITS)+"b}").format(self.pwr_mask >> NUM_SIGBITS),
        #     "| signal :", ("{0:0"+str(NUM_SIGBITS*2)+"b}").format(sig),
        #     "| ACT Result :", ("{0:0"+str(NUM_SIGBITS)+"b}").format(self.act_mask & sig),
        #     "| bits count :", self.count(self.act_mask & sig),
        #     "| ACTIVATE? :", self.count(self.act)
        # )


    def isActivate(self):
        return self.act

    # we only run this.
    def apply(self, sig):
        self.compare(sig)
        if self.isActivate() :
            self.sav_sig = sig
            '''
            print("ACTIVATE :", ("{0:0"+str(NUM_SIGBITS*2)+"b}").format(self.act_mask), "SIGNAL :", ("{0:0"+str(NUM_SIGBITS*2)+"b}").format(sig))
        else :
            print("_UNACTED :", ("{0:0"+str(NUM_SIGBITS*2)+"b}").format(self.act_mask), "SIGNAL :", ("{0:0"+str(NUM_SIGBITS*2)+"b}").format(sig))
            '''

    # we will implement later about pure cell
    # so power function should be removed as soon as implementation.
    def power(self) :
        for i in range(len(SIGNALS)) :
            if self.count((self.pwr_mask & self.sav_sig) & (SIGNALS[i]<<NUM_SIGBITS)) >= NUM_ACVIVATED :
                return i
        return 0
