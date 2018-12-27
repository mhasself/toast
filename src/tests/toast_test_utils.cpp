
// Copyright (c) 2015-2019 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#include <toast_test.hpp>

#include <thread>
#include <chrono>


TEST_F(TOASTutilsTest, logging) {
    auto & env = toast::Environment::get();

    std::cout << "Testing level CRITICAL" << std::endl;
    env.set_log_level("CRITICAL");
    auto & log = toast::Logger::get();
    log.critical("This message level is CRITICAL");
    log.error("This message level is ERROR");
    log.warning("This message level is WARNING");
    log.info("This message level is INFO");
    log.debug("This message level is DEBUG");

    std::cout << "Testing level ERROR" << std::endl;
    env.set_log_level("ERROR");
    log = toast::Logger::get();
    log.critical("This message level is CRITICAL");
    log.error("This message level is ERROR");
    log.warning("This message level is WARNING");
    log.info("This message level is INFO");
    log.debug("This message level is DEBUG");

    std::cout << "Testing level WARNING" << std::endl;
    env.set_log_level("WARNING");
    log = toast::Logger::get();
    log.critical("This message level is CRITICAL");
    log.error("This message level is ERROR");
    log.warning("This message level is WARNING");
    log.info("This message level is INFO");
    log.debug("This message level is DEBUG");

    std::cout << "Testing level INFO" << std::endl;
    env.set_log_level("INFO");
    log = toast::Logger::get();
    log.critical("This message level is CRITICAL");
    log.error("This message level is ERROR");
    log.warning("This message level is WARNING");
    log.info("This message level is INFO");
    log.debug("This message level is DEBUG");

    std::cout << "Testing level DEBUG" << std::endl;
    env.set_log_level("DEBUG");
    log = toast::Logger::get();
    log.critical("This message level is CRITICAL");
    log.error("This message level is ERROR");
    log.warning("This message level is WARNING");
    log.info("This message level is INFO");
    log.debug("This message level is DEBUG");
}

TEST_F(TOASTutilsTest, singletimer) {
    int incr = 200;
    double dincr = (double)incr / 1000.0;
    double prec = 1.0e-2;
    toast::Timer tm;
    EXPECT_EQ(false, tm.is_running());
    tm.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(incr));
    tm.stop();
    ASSERT_NEAR(dincr, tm.seconds(), prec);
    tm.report("Test timer stopped");
    tm.clear();
    tm.start();
    try {
        tm.report("This should throw since timer not stopped...");
    } catch (toast::Exception & e) {
        std::cout << "This should throw since timer not stopped..."
                  << std::endl;
        std::cout << e.what() << std::endl;
    }
    EXPECT_EQ(true, tm.is_running());
    tm.stop();
}


TEST_F(TOASTutilsTest, globaltimer) {
    int incr = 200;
    double dincr = (double)incr / 1000.0;
    double prec = 1.0e-2;
    auto & gtm = toast::GlobalTimers::get();

    std::vector <std::string> tnames = {
        "timer1",
        "timer2",
        "timer3"
    };

    for (auto const & tname : tnames) {
        try {
            gtm.stop(tname);
        } catch (toast::Exception & e) {
            std::cout << "This should throw since timer " << tname
                      << " not yet created" << std::endl;
            std::cout << e.what() << std::endl;
        }
    }

    for (auto const & tname : tnames) {
        gtm.start(tname);
    }

    for (auto const & tname : tnames) {
        EXPECT_EQ(true, gtm.is_running(tname));
        try {
            gtm.stop(tname);
        } catch (toast::Exception & e) {
            std::cout << "This should throw since timer " << tname
                      << " still running" << std::endl;
            std::cout << e.what() << std::endl;
        }
    }

    gtm.stop_all();
    gtm.clear_all();
    for (auto const & tname : tnames) {
        gtm.start(tname);
    }

    for (auto const & tname : tnames) {
        std::this_thread::sleep_for(std::chrono::milliseconds(incr));
        gtm.stop(tname);
    }

    size_t offset = 1;
    for (auto const & tname : tnames) {
        ASSERT_NEAR((double)offset * dincr, gtm.seconds(tname), prec);
        offset++;
    }

    gtm.report();
    gtm.clear_all();

    for (auto const & tname : tnames) {
        gtm.start(tname);
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(incr));

    gtm.stop_all();
    for (auto const & tname : tnames) {
        ASSERT_NEAR(dincr, gtm.seconds(tname), prec);
    }

    gtm.report();
}
